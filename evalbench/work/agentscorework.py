"""AgentScoreWork class."""

from typing import Any
from work.work import Work
from scorers import score as scorer
import threading


class AgentScoreWork(Work):
    """Work class for scoring agent generation results."""

    def __init__(
        self,
        config: dict,
        eval_output: dict,
        scoring_results: list,
        global_models: Any = None,
    ):
        self.config = config
        self.eval_output = eval_output
        self.scoring_results = scoring_results

        if global_models is None:
            global_models = {
                "lock": threading.Lock(),
                "semaphores": {},
                "registered_models": {}
            }
        self.global_models = global_models

    def run(self, work_config: Any = None) -> Any:
        """Runs the agent scoring work.

        Args:
            work_config: Optional configuration for the work.

        Returns:
            The scoring result dictionary.
        """
        scenario = self.eval_output.get("scenario", {})
        metadata = self.eval_output.get("metadata", {})
        golden_sql = self.eval_output.get("golden_sql", "")
        generated_sql = self.eval_output.get("generated_sql", "")
        turn_history = self.eval_output.get("turn_history", [])
        if not golden_sql and turn_history:
            for turn_item in turn_history:
                if turn_item.get("golden_sql"):
                    golden_sql = turn_item["golden_sql"]
                    break
        if not generated_sql and turn_history:
            for turn_item in reversed(turn_history):
                if turn_item.get("generated_sql"):
                    generated_sql = turn_item["generated_sql"]
                    break
        golden_result = self.eval_output.get("golden_result")
        if golden_result is None:
            golden_result = scenario.get("expected_trajectory", [])
        generated_result = self.eval_output.get("generated_result")
        if generated_result is None:
            generated_result = self.eval_output.get("accumulated_tools", [])

        item_id = self.eval_output.get("id") or self.eval_output.get("eval_id") or ""
        item_prompt = self.eval_output.get("nl_prompt") or self.eval_output.get("prompt") or scenario.get("starting_prompt", "")

        other = self.eval_output.get("other", {}) if isinstance(self.eval_output.get("other"), dict) else {}
        scenario_other = scenario.get("other", {}) if isinstance(scenario.get("other"), dict) else {}

        is_ambiguous = bool(
            scenario.get("is_ambiguous", False)
            or self.eval_output.get("is_ambiguous", False)
            or other.get("is_ambiguous", False)
            or scenario_other.get("is_ambiguous", False)
            or (not golden_sql and not golden_result)
        )

        generated_disambig = (
            self.eval_output.get("generated_disambiguation_question")
            or other.get("generated_disambiguation_question")
            or other.get("disambiguation_question")
            or self.eval_output.get("disambiguation_question")
            or (other.get("is_disambiguation") in ("true", True))
        )

        scoring_item = {
            "id": item_id,
            "nl_prompt": item_prompt,
            "golden_sql": golden_sql,
            "query_type": "disambiguation" if is_ambiguous else "dql",
            "is_ambiguous": is_ambiguous,
            "generated_disambiguation_question": generated_disambig,
            "other": other,
            "golden_result": golden_result,
            "golden_eval_results": "",
            "golden_error": self.eval_output.get("golden_error", ""),
            "generated_sql": generated_sql if generated_sql else "skipped",
            "generated_result": generated_result,
            "eval_results": self.eval_output,
            "generated_error": self.eval_output.get("generated_error"),
            "dialects": metadata.get("dialects", []),
            "database": metadata.get("database", "unknown"),
            "job_id": self.eval_output.get("job_id"),
            "turn_history": turn_history,
            "accumulated_tools": self.eval_output.get("accumulated_tools", []),
            "accumulated_skills": self.eval_output.get("accumulated_skills", []),
        }

        scorer.compare(
            eval_output_item=scoring_item,
            experiment_config=self.config,
            scoring_results=self.scoring_results,
            global_models=self.global_models
        )

        base_item = {
            "id": item_id,
            "generated_sql": generated_sql if generated_sql else "skipped",
            "generated_error": self.eval_output.get("generated_error"),
            "dialects": metadata.get("dialects", []),
            "database": metadata.get("database", "unknown"),
            "job_id": self.eval_output.get("job_id"),
            "comparison_logs": None,
            "comparison_error": None,
        }

        # Multi-turn rollup metrics calculation
        if turn_history:
            sql_turns = [t for t in turn_history if t.get("golden_sql") or t.get("generated_sql")]
            if sql_turns:
                set_match_scores = [t.get("set_match", 0.0) for t in sql_turns]
                all_turns_score = 100.0 if all(s == 100.0 for s in set_match_scores) else 0.0
                mean_score = sum(set_match_scores) / len(set_match_scores)

                # Record multi-turn aggregate metrics
                self.scoring_results.append({
                    **base_item,
                    "comparator": "set_match_all_turns",
                    "score": all_turns_score,
                })
                self.scoring_results.append({
                    **base_item,
                    "comparator": "set_match_mean",
                    "score": mean_score,
                })

                for t_idx, t in enumerate(turn_history):
                    if "set_match" in t:
                        self.scoring_results.append({
                            **base_item,
                            "comparator": f"set_match_turn_{t_idx + 1}",
                            "score": float(t["set_match"]),
                        })

        # Calculate 3-Tier Composite Score for Brewmax Parity:
        # Composite = 0.90 * Content + 0.05 * Conciseness + 0.05 * BestPractices
        eval_id = self.eval_output.get("eval_id")
        current_scores = [r for r in self.scoring_results if r.get("id") == eval_id]
        content_score = 0.0
        for comp_name in ["llmrater", "goal_completion", "set_match_all_turns", "set_match"]:
            found = False
            for r in current_scores:
                if r.get("comparator") == comp_name and r.get("score") is not None:
                    content_score = float(r["score"])
                    found = True
                    break
            if found:
                break

        accumulated_tools = self.eval_output.get("accumulated_tools", [])
        tool_counts = {}
        for t in accumulated_tools:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        has_excessive_duplicates = any(c > 4 for c in tool_counts.values())
        conciseness_score = 50.0 if has_excessive_duplicates else 100.0
        best_practices_score = 100.0

        composite_score = round(
            0.90 * content_score + 0.05 * conciseness_score + 0.05 * best_practices_score, 2
        )

        self.scoring_results.append({
            **base_item,
            "comparator": "composite_score",
            "score": composite_score,
            "comparison_logs": f"Content={content_score} (0.90), Conciseness={conciseness_score} (0.05), BestPractices={best_practices_score} (0.05)",
        })

        return self.eval_output
