"""Rescore tool for EvalBench: Offline re-evaluation and calibration of saved traces.

Allows rapid rater calibration, prompt template testing, and scoring parity experiments
without re-running agent generation or BigQuery query executions.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import logging
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional
import pandas as pd

import databases
from reporting import analyzer
from reporting.report import STORETYPE
from scorers.setmatcher import SetMatcher
from util.config import load_yaml_config, set_session_configs
from util.service import load_session_configs
from work.agentscorework import AgentScoreWork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _safe_parse(val: Any, default: Any = None) -> Any:
    """Safely parse JSON or python literal strings, returning default if empty."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, (dict, list)):
        return val
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ("none", "null", "nan"):
        return default
    try:
        return json.loads(val_str)
    except Exception:
        pass
    try:
        return ast.literal_eval(val_str)
    except Exception:
        pass
    return val_str


def _extract_telemetry_sql(row: dict, scenario: dict, turn_history: list) -> str:
    """Extracts SQL from internal telemetry/actions when top-level generated_sql is blank."""
    # 1. Check turn history
    if turn_history:
        for t in reversed(turn_history):
            if isinstance(t, dict):
                sql = t.get("generated_sql") or t.get("sql") or t.get("query")
                if sql and isinstance(sql, str) and sql.strip() and sql.strip() != "skipped":
                    return sql.strip()

    # 2. Check sources for actions/telemetry
    sources = [
        row.get("macchiato_debug_info"),
        row.get("debug_info"),
        row.get("other"),
        scenario.get("other"),
        scenario.get("macchiato_debug_info"),
    ]

    def _find_sql_in_obj(obj: Any) -> Optional[str]:
        if not obj:
            return None
        parsed = _safe_parse(obj)
        if isinstance(parsed, dict):
            for k in ("sql", "query", "sql_query", "executed_sql"):
                if k in parsed and isinstance(parsed[k], str) and parsed[k].strip():
                    return parsed[k].strip()
            if "input" in parsed and isinstance(parsed["input"], dict):
                for k in ("sql", "query", "sql_query"):
                    if k in parsed["input"] and isinstance(parsed["input"][k], str) and parsed["input"][k].strip():
                        return parsed["input"][k].strip()
            for list_k in ("actions", "tool_calls", "tools"):
                if list_k in parsed:
                    res = _find_sql_in_obj(parsed[list_k])
                    if res:
                        return res
            for k, v in parsed.items():
                if isinstance(v, (dict, list)):
                    res = _find_sql_in_obj(v)
                    if res:
                        return res
        elif isinstance(parsed, list):
            for item in reversed(parsed):
                res = _find_sql_in_obj(item)
                if res:
                    return res
        return None

    for src in sources:
        found_sql = _find_sql_in_obj(src)
        if found_sql:
            return found_sql

    return ""


def _clean_sql_query(query: str) -> str:
    """Strips markdown code blocks, Sherlog trace headers, and surrounding whitespace."""
    if not query or not isinstance(query, str):
        return ""
    q = query.strip()
    if not q or q.lower() == "skipped":
        return ""

    # 1. Strip markdown code fences (e.g. ```sql ... ``` or ``` ...)
    fence_pattern = re.compile(r"```(?:sql)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(q)
    if match:
        q = match.group(1).strip()
    elif q.startswith("```"):
        q = re.sub(r"^```(?:sql)?\s*", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s*```$", "", q)
        q = q.strip()

    # 2. Strip leading comments or trace annotations (e.g. -- [Sherlog Trace] or multiple comment lines)
    lines = q.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-- [Sherlog Trace]") or stripped.startswith("-- [Telemetry]"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _get_cached_db(
    database_name: str,
    dialect: str,
    db_configs: Any,
    db_cache: dict,
    db_lock: threading.Lock,
) -> Optional[Any]:
    """Retrieves or creates a thread-safe database connection instance from cache."""
    cache_key = f"{dialect}_{database_name}"
    with db_lock:
        if cache_key in db_cache:
            return db_cache[cache_key]

        db_cfg = None
        if isinstance(db_configs, dict):
            dialect_configs = db_configs.get(dialect)
            if isinstance(dialect_configs, list) and dialect_configs:
                db_cfg = dialect_configs[0]
            elif isinstance(dialect_configs, dict):
                db_cfg = dialect_configs
        elif isinstance(db_configs, list) and db_configs:
            db_cfg = db_configs[0]

        if not db_cfg:
            db_cfg = {"db_type": dialect or "bigquery"}

        db_cfg_copy = db_cfg.copy() if isinstance(db_cfg, dict) else {"db_type": dialect or "bigquery"}
        if "db_type" not in db_cfg_copy:
            db_cfg_copy["db_type"] = dialect or "bigquery"

        try:
            db = databases.get_database(db_cfg_copy, database_name)
            db_cache[cache_key] = db
            return db
        except Exception as e:
            logging.warning(
                f"Could not initialize database '{database_name}' for dialect '{dialect}': {e}"
            )
            return None


def _reexecute_single_sql(db: Any, sql_query: str) -> tuple[Optional[List[Any]], Optional[str]]:
    """Executes a single SQL query against db, returning (result_rows, error_str)."""
    cleaned = _clean_sql_query(sql_query)
    if not cleaned or cleaned.lower() == "skipped":
        return None, None
    try:
        res, _, err = db.execute(cleaned, use_cache=True, rollback=True)
        return (res if res is not None else []), (str(err) if err else None)
    except Exception as e:
        return [], str(e)


def _reexecute_trace_sql(
    eval_output: dict,
    db_configs: Any,
    db_cache: dict,
    db_lock: threading.Lock,
    set_matcher: Optional[Any] = None,
    force: bool = False,
    only_on_empty: bool = False,
) -> None:
    """Re-executes generated and golden SQL against target database and refreshes turn set_match."""
    metadata = eval_output.get("metadata", {})
    scenario = eval_output.get("scenario", {})
    database_name = (
        eval_output.get("database")
        or metadata.get("database")
        or scenario.get("database", "")
    )
    dialects = (
        metadata.get("dialects")
        or scenario.get("dialects", ["bigquery"])
    )
    dialect = dialects[0] if isinstance(dialects, list) and dialects else "bigquery"

    db = _get_cached_db(database_name, dialect, db_configs, db_cache, db_lock)
    if not db:
        return

    # 1. Top-Level Generated SQL Re-Execution
    gen_sql = eval_output.get("generated_sql", "")
    cleaned_gen_sql = _clean_sql_query(gen_sql)
    if cleaned_gen_sql:
        eval_output["generated_sql"] = cleaned_gen_sql
        should_run_gen = force or (
            only_on_empty
            and (
                eval_output.get("generated_result") is None
                or eval_output.get("generated_result") == []
                or eval_output.get("generated_error")
            )
        )
        if should_run_gen:
            res, err = _reexecute_single_sql(db, cleaned_gen_sql)
            eval_output["generated_result"] = res if res is not None else []
            eval_output["generated_error"] = err

    # 2. Top-Level Golden SQL Re-Execution
    gold_sql = eval_output.get("golden_sql", "")
    cleaned_gold_sql = _clean_sql_query(gold_sql)
    if cleaned_gold_sql:
        eval_output["golden_sql"] = cleaned_gold_sql
        should_run_gold = force or (
            only_on_empty
            and (
                eval_output.get("golden_result") is None
                or eval_output.get("golden_result") == []
                or eval_output.get("golden_error")
            )
        )
        if should_run_gold:
            res, err = _reexecute_single_sql(db, cleaned_gold_sql)
            eval_output["golden_result"] = res if res is not None else []
            eval_output["golden_error"] = err

    # 3. Multi-Turn Trajectory Awareness (turn_history re-execution & set_match refresh)
    turn_history = eval_output.get("turn_history", [])
    if isinstance(turn_history, list):
        for turn_item in turn_history:
            if not isinstance(turn_item, dict):
                continue
            t_gen_sql = _clean_sql_query(turn_item.get("generated_sql") or turn_item.get("sql") or "")
            t_gold_sql = _clean_sql_query(turn_item.get("golden_sql") or "")

            if t_gen_sql:
                turn_item["generated_sql"] = t_gen_sql
                should_run_t_gen = force or (
                    only_on_empty
                    and (
                        turn_item.get("generated_execution_result") is None
                        or turn_item.get("generated_execution_result") == []
                        or turn_item.get("generated_error")
                    )
                )
                if should_run_t_gen:
                    res, err = _reexecute_single_sql(db, t_gen_sql)
                    turn_item["generated_execution_result"] = res if res is not None else []
                    turn_item["generated_error"] = err

            if t_gold_sql:
                turn_item["golden_sql"] = t_gold_sql
                should_run_t_gold = force or (
                    only_on_empty
                    and (
                        turn_item.get("golden_execution_result") is None
                        or turn_item.get("golden_execution_result") == []
                        or turn_item.get("golden_error")
                    )
                )
                if should_run_t_gold:
                    res, err = _reexecute_single_sql(db, t_gold_sql)
                    turn_item["golden_execution_result"] = res if res is not None else []
                    turn_item["golden_error"] = err

            if set_matcher and (t_gen_sql or t_gold_sql):
                try:
                    score, _ = set_matcher.compare(
                        nl_prompt=turn_item.get("user_prompt", ""),
                        golden_query=t_gold_sql,
                        query_type="dql",
                        golden_execution_result=turn_item.get("golden_execution_result") or [],
                        golden_eval_result="",
                        golden_error=str(turn_item.get("golden_error") or ""),
                        generated_query=t_gen_sql,
                        generated_execution_result=turn_item.get("generated_execution_result") or [],
                        generated_eval_result="",
                        generated_error=str(turn_item.get("generated_error") or ""),
                    )
                    turn_item["set_match"] = float(score)
                except Exception as e:
                    logging.warning(
                        f"SetMatcher comparison failed on turn {turn_item.get('turn')}: {e}"
                    )
                    turn_item["set_match"] = 0.0


def _row_to_eval_output(row: dict) -> dict:
    """Converts a row from evals.csv / results.json into an eval_output dictionary."""
    scenario = _safe_parse(row.get("scenario"), {})
    if not isinstance(scenario, dict):
        scenario = {}

    conversation_history = _safe_parse(row.get("conversation_history"), [])
    turn_history = _safe_parse(row.get("turn_history"), [])
    metadata = _safe_parse(row.get("metadata"), {})
    if not isinstance(metadata, dict):
        metadata = {}

    other = _safe_parse(row.get("other"), {})
    if not isinstance(other, dict):
        other = {}

    is_ambiguous = (
        row.get("is_ambiguous")
        if pd.notna(row.get("is_ambiguous")) and str(row.get("is_ambiguous")).strip() != ""
        else (scenario.get("is_ambiguous") or other.get("is_ambiguous") or metadata.get("is_ambiguous", False))
    )
    if isinstance(is_ambiguous, str):
        is_ambiguous = is_ambiguous.lower() in ("true", "1", "t", "yes")
    else:
        is_ambiguous = bool(is_ambiguous)

    generated_disambig = (
        row.get("generated_disambiguation_question")
        or row.get("disambiguation_question")
        or other.get("generated_disambiguation_question")
        or other.get("disambiguation_question")
        or scenario.get("generated_disambiguation_question")
    )

    accumulated_tools = _safe_parse(row.get("accumulated_tools"), [])
    if not isinstance(accumulated_tools, list):
        accumulated_tools = []

    accumulated_skills = _safe_parse(row.get("accumulated_skills"), [])
    if not isinstance(accumulated_skills, list):
        accumulated_skills = []

    item_id = str(row.get("id") or row.get("eval_id") or scenario.get("id") or "")
    prompt = str(
        row.get("nl_prompt")
        or row.get("prompt")
        or scenario.get("starting_prompt")
        or ""
    )

    generated_sql = str(row.get("generated_sql") or "").strip()
    if not generated_sql or generated_sql == "skipped":
        telemetry_sql = _extract_telemetry_sql(row, scenario, turn_history if isinstance(turn_history, list) else [])
        if telemetry_sql:
            generated_sql = telemetry_sql

    generated_result = _safe_parse(row.get("generated_result"))
    golden_result = _safe_parse(row.get("golden_result"))

    eval_output = {
        "id": item_id,
        "eval_id": item_id,
        "nl_prompt": prompt,
        "prompt": prompt,
        "stdout": str(row.get("stdout") or ""),
        "stderr": str(row.get("stderr") or ""),
        "returncode": int(row.get("returncode", 0)) if pd.notna(row.get("returncode")) else 0,
        "prompt_generator_error": row.get("prompt_generator_error") if pd.notna(row.get("prompt_generator_error")) else None,
        "generated_error": str(row.get("generated_error")) if pd.notna(row.get("generated_error")) and str(row.get("generated_error")).strip() != "" else None,
        "sql_generator_error": row.get("sql_generator_error") if pd.notna(row.get("sql_generator_error")) else None,
        "golden_error": str(row.get("golden_error")) if pd.notna(row.get("golden_error")) and str(row.get("golden_error")).strip() != "" else None,
        "generated_sql": generated_sql if generated_sql else "skipped",
        "golden_sql": str(row.get("golden_sql") or ""),
        "generated_result": generated_result if generated_result is not None else accumulated_tools,
        "golden_result": golden_result if golden_result is not None else scenario.get("expected_trajectory", []),
        "conversation_history": json.dumps(conversation_history, indent=2) if isinstance(conversation_history, list) else str(conversation_history or ""),
        "turn_history": turn_history if isinstance(turn_history, list) else [],
        "scenario": scenario,
        "accumulated_tools": accumulated_tools,
        "accumulated_skills": accumulated_skills,
        "job_id": str(row.get("job_id") or "rescored_job"),
        "metadata": metadata,
        "other": other,
        "is_ambiguous": is_ambiguous,
        "generated_disambiguation_question": generated_disambig,
    }
    return eval_output


def load_traces(results_file: str) -> List[dict]:
    """Loads traces from an evals.csv, results.json, or directory path."""
    if os.path.isdir(results_file):
        csv_candidate = os.path.join(results_file, "evals.csv")
        json_candidate = os.path.join(results_file, "results.json")
        if os.path.exists(csv_candidate):
            results_file = csv_candidate
        elif os.path.exists(json_candidate):
            results_file = json_candidate
        else:
            raise FileNotFoundError(f"No evals.csv or results.json found in directory '{results_file}'")

    logging.info(f"Loading traces from: {results_file}")
    if results_file.endswith(".csv"):
        df = pd.read_csv(results_file)
        records = df.to_dict(orient="records")
    elif results_file.endswith(".json"):
        with open(results_file, "r") as f:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported file format for '{results_file}'. Use .csv or .json")

    eval_outputs = [_row_to_eval_output(r) for r in records]
    logging.info(f"Successfully loaded {len(eval_outputs)} scenario traces.")
    return eval_outputs


def rescore(
    results_file: str,
    config_file: str,
    output_dir: Optional[str] = None,
    workers: int = 20,
    scenarios: Optional[List[str]] = None,
    limit: Optional[int] = None,
    reexecute_sql: bool = False,
    reexecute_on_empty: bool = False,
    save_refreshed_evals: bool = False,
    prompt_template: Optional[str] = None,
) -> pd.DataFrame:
    """Executes offline rescoring across all traces in parallel."""
    # 1. Load Configurations
    parsed_config = load_yaml_config(config_file)
    session = {}
    set_session_configs(session, parsed_config)
    config, db_configs, model_config, setup_config = load_session_configs(session)

    if prompt_template:
        scorers_cfg = config.setdefault("scorers", {})
        if "llmrater" in scorers_cfg and isinstance(scorers_cfg["llmrater"], dict):
            scorers_cfg["llmrater"]["prompt_template"] = prompt_template
        else:
            scorers_cfg["llmrater"] = {"prompt_template": prompt_template}
        logging.info(f"Overriding LLM rater prompt_template: '{prompt_template}'")

    # 2. Load Traces
    traces = load_traces(results_file)

    # 3. Filter traces if requested
    if scenarios:
        scenario_set = set(str(s).strip() for s in scenarios)
        traces = [t for t in traces if str(t.get("id")) in scenario_set or str(t.get("eval_id")) in scenario_set]
        logging.info(f"Filtered to {len(traces)} matching scenarios.")

    if limit and limit > 0:
        traces = traces[:limit]
        logging.info(f"Limiting to first {len(traces)} traces.")

    if not traces:
        logging.warning("No traces found to rescore.")
        return pd.DataFrame()

    job_id = traces[0].get("job_id", "rescored_run")
    run_time = datetime.datetime.now().isoformat()

    # 4. Optional Database SQL Re-Execution
    if reexecute_sql or reexecute_on_empty:
        logging.info(
            f"Starting database query re-execution (force={reexecute_sql}, only_on_empty={reexecute_on_empty}) across {len(traces)} traces..."
        )
        set_matcher_cfg = config.get("scorers", {}).get("set_match", {})
        set_matcher = SetMatcher(set_matcher_cfg) if set_matcher_cfg is not None else SetMatcher({})
        db_cache: dict = {}
        db_lock = threading.Lock()

        db_start_time = datetime.datetime.now()
        with ThreadPoolExecutor(max_workers=workers) as db_executor:
            futures = {
                db_executor.submit(
                    _reexecute_trace_sql,
                    trace,
                    db_configs,
                    db_cache,
                    db_lock,
                    set_matcher,
                    force=reexecute_sql,
                    only_on_empty=reexecute_on_empty,
                ): trace
                for trace in traces
            }
            completed_db = 0
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    trace_id = futures[future].get("id")
                    logging.error(f"Error re-executing SQL for trace '{trace_id}': {e}", exc_info=True)
                completed_db += 1
                if completed_db % 50 == 0 or completed_db == len(traces):
                    elapsed = (datetime.datetime.now() - db_start_time).total_seconds()
                    logging.info(f"DB Re-execution Progress: [{completed_db}/{len(traces)}] completed ({elapsed:.1f}s elapsed)")

        db_duration = (datetime.datetime.now() - db_start_time).total_seconds()
        logging.info(f"Database query re-execution completed in {db_duration:.2f}s.")

    # 5. Setup Parallel Rescoring
    global_models = {
        "lock": threading.Lock(),
        "semaphores": {},
        "registered_models": {},
    }

    all_scores: List[dict] = []
    scores_lock = threading.Lock()

    def _score_single_trace(eval_output: dict):
        local_results: List[dict] = []
        score_work = AgentScoreWork(
            config=config,
            eval_output=eval_output,
            scoring_results=local_results,
            global_models=global_models,
        )
        score_work.run()
        with scores_lock:
            all_scores.extend(local_results)

    logging.info(f"Starting parallel rescoring across {len(traces)} traces using {workers} threads...")
    start_time = datetime.datetime.now()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_score_single_trace, trace): trace for trace in traces}
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                trace_id = futures[future].get("id")
                logging.error(f"Error rescoring trace '{trace_id}': {e}", exc_info=True)
            completed += 1
            if completed % 50 == 0 or completed == len(traces):
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                logging.info(f"Progress: [{completed}/{len(traces)}] completed ({elapsed:.1f}s elapsed)")

    duration = (datetime.datetime.now() - start_time).total_seconds()
    logging.info(f"Rescoring finished in {duration:.2f}s ({len(all_scores)} score records generated).")

    # 6. Analyze Results
    scores_df, summary_scores_df = analyzer.analyze_result(
        scores=all_scores,
        experiment_config=config,
        num_prompts=len(traces),
        num_trials=1,
    )
    summary_scores_df["job_id"] = job_id
    summary_scores_df["run_time"] = run_time

    # 7. Save Artifacts
    if not output_dir:
        base_dir = os.path.dirname(os.path.abspath(results_file))
        output_dir = os.path.join(base_dir, "rescored")

    os.makedirs(output_dir, exist_ok=True)
    scores_csv_path = os.path.join(output_dir, "scores.csv")
    summary_csv_path = os.path.join(output_dir, "summary.csv")

    scores_df.to_csv(scores_csv_path, index=False)
    summary_scores_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Saved rescored scores to: {scores_csv_path}")
    logging.info(f"Saved rescored summary to: {summary_csv_path}")

    if save_refreshed_evals or reexecute_sql or reexecute_on_empty:
        refreshed_rows = []
        for t in traces:
            row_dict = dict(t)
            for k in ("turn_history", "scenario", "metadata", "generated_result", "golden_result", "accumulated_tools", "accumulated_skills"):
                if isinstance(row_dict.get(k), (dict, list)):
                    row_dict[k] = json.dumps(row_dict[k])
            refreshed_rows.append(row_dict)
        refreshed_df = pd.DataFrame(refreshed_rows)
        refreshed_csv_path = os.path.join(output_dir, "evals_refreshed.csv")
        refreshed_df.to_csv(refreshed_csv_path, index=False)
        logging.info(f"Saved refreshed execution traces to: {refreshed_csv_path}")

    # 8. Print Terminal Scorecard Table
    print("\n" + "=" * 65)
    print(f"       EVALBENCH OFFLINE RESCORE SCORECARD (N={len(traces)})")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Score / Rate':<15} | {'Count / Total'}")
    print("-" * 65)
    for _, row in summary_scores_df.iterrows():
        name = str(row.get("metric_name", ""))
        pct = row.get("percentage_score")
        correct = row.get("correct_results_count", "")
        total = row.get("original_df_size", "")
        pct_str = f"{pct:.2f}%" if pd.notna(pct) and isinstance(pct, (int, float)) else str(pct)
        count_str = f"{correct} / {total}" if correct != "" and total != "" else ""
        print(f"{name:<30} | {pct_str:<15} | {count_str}")
    print("=" * 65 + "\n")

    return summary_scores_df


def main():
    parser = argparse.ArgumentParser(
        description="EvalBench Offline Rescoring CLI: Calibrate raters without re-running agent execution."
    )
    parser.add_argument(
        "--results_file",
        "-r",
        required=True,
        help="Path to evals.csv, results.json, or directory containing execution traces.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment run configuration YAML.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        help="Directory to save rescored scores.csv and summary.csv.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=20,
        help="Number of concurrent worker threads for scoring (default: 20).",
    )
    parser.add_argument(
        "--scenarios",
        "-s",
        nargs="+",
        default=None,
        help="Optional list of scenario IDs to rescore.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Optional limit on number of scenarios to rescore.",
    )
    parser.add_argument(
        "--reexecute_sql",
        "-x",
        action="store_true",
        default=False,
        help="Force re-execute all generated and golden SQL queries against target database before scoring.",
    )
    parser.add_argument(
        "--reexecute_on_empty",
        action="store_true",
        default=False,
        help="Re-execute SQL queries against target database only where execution results are empty or have errors.",
    )
    parser.add_argument(
        "--save_refreshed_evals",
        action="store_true",
        default=False,
        help="Save refreshed execution traces and turn histories to evals_refreshed.csv in output directory.",
    )

    parser.add_argument(
        "--prompt_template",
        "-p",
        default=None,
        help="Optional prompt template override for LLM rater (e.g. 'brewmax' or 'default').",
    )

    args = parser.parse_args()
    rescore(
        results_file=args.results_file,
        config_file=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        scenarios=args.scenarios,
        limit=args.limit,
        reexecute_sql=args.reexecute_sql,
        reexecute_on_empty=args.reexecute_on_empty,
        save_refreshed_evals=args.save_refreshed_evals,
        prompt_template=args.prompt_template,
    )


if __name__ == "__main__":
    main()
