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
import sys
import threading
from typing import Any, Dict, List, Optional
import pandas as pd

from reporting import analyzer
from reporting.report import STORETYPE
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
        "generated_sql": str(row.get("generated_sql") or "skipped"),
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
) -> pd.DataFrame:
    """Executes offline rescoring across all traces in parallel."""
    # 1. Load Configurations
    parsed_config = load_yaml_config(config_file)
    session = {}
    set_session_configs(session, parsed_config)
    config, db_configs, model_config, setup_config = load_session_configs(session)

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

    # 4. Setup Parallel Rescoring
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

    # 5. Analyze Results
    scores_df, summary_scores_df = analyzer.analyze_result(
        scores=all_scores,
        experiment_config=config,
        num_prompts=len(traces),
        num_trials=1,
    )
    summary_scores_df["job_id"] = job_id
    summary_scores_df["run_time"] = run_time

    # 6. Save Artifacts
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

    # 7. Print Terminal Scorecard Table
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

    args = parser.parse_args()
    rescore(
        results_file=args.results_file,
        config_file=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        scenarios=args.scenarios,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
