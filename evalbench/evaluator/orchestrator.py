import datetime
import json
from typing import Any, List, Dict
import uuid

from dataset.evalinput import EvalInputRequest


def sanitize_eval_output(item: Any, max_rows: int = 50) -> Any:
    """Truncates large result sets and telemetry arrays to prevent memory explosion during JSON serialization."""
    if not isinstance(item, dict):
        return item

    sanitized = dict(item)

    # 1. Truncate top-level result lists
    for key in ("generated_result", "golden_result", "accumulated_tools", "accumulated_skills"):
        val = sanitized.get(key)
        if isinstance(val, list) and len(val) > max_rows:
            sanitized[key] = val[:max_rows]

    # 2. Truncate turn_history tables
    turn_history = sanitized.get("turn_history")
    if isinstance(turn_history, list):
        sanitized_turns = []
        for t in turn_history:
            if isinstance(t, dict):
                t_copy = dict(t)
                for t_key in ("generated_execution_result", "golden_execution_result", "tools"):
                    t_val = t_copy.get(t_key)
                    if isinstance(t_val, list) and len(t_val) > max_rows:
                        t_copy[t_key] = t_val[:max_rows]
                sanitized_turns.append(t_copy)
            else:
                sanitized_turns.append(t)
        sanitized["turn_history"] = sanitized_turns

    return sanitized


def dump_compact_json(data: Any, temp_file) -> str:
    """Dumps data to a temporary file using compact separators and returns the file path."""
    json.dump(data, temp_file, sort_keys=True, separators=(",", ":"), default=str)
    return temp_file.name


# The `Orchestrator` class is a Python class that serves as a central component for
# orchestrating the evaluation process of datasets. It initializes with various configurations
# and settings, generates a unique job ID, and keeps track of evaluation outputs and scoring
# results. The class has methods for evaluating datasets, breaking down evaluations by
# categories, and processing the evaluation results. It also has attributes for managing the
# number of evaluation runners and SQL execution runners. The `evaluate` method is a wrapper
# that handles the evaluation process by category, while the `evaluate_sub_dataset` method is
# responsible for evaluating sub-datasets with specific database configurations. The `process`
# method is likely intended to execute the evaluation process.
class Orchestrator:
    def __init__(
        self,
        config,
        db_configs,
        setup_config,
        report_progress=False,
    ):
        self.config = config
        self.db_configs = db_configs
        self.setup_config = setup_config
        self.job_id = f"{uuid.uuid4()}"
        self.run_time = datetime.datetime.now()
        self.total_eval_outputs = []
        self.total_scoring_results = []
        self.reporting_total_evals_done = 0
        self.report_progress = report_progress

        runner_config = self.config.get("runners", {})
        self.eval_runners = runner_config.get("eval_runners", 4)
        self.sqlexec_runners = runner_config.get("sqlexec_runners", 10)

    def evaluate(self, dataset: list[EvalInputRequest]):
        """This wrapper breaks down evaluations by category of evaluations. (dql, dml, ddl).
        This allows the module to prepare the correct database connections as DDL queries
        require setting up and tearing down the databsae and DML queries require prevention
        of unintended consequences. Additionally, DQLs are run under a read-only user.
        """

    def evaluate_sub_dataset(
        self,
        sub_datasets,
        db_config,
        dialect,
        database,
        progress_reporting,
        global_models,
    ):
        pass

    def process(self):
        return (
            self.job_id,
            self.run_time,
            None,
            None,
            None,
        )

    def get_display_dataset_config(self) -> str:
        """Returns a sanitized configuration name suitable for safe log outputs."""
        cand = self.config.get("dataset_config", "Unknown config")
        if not isinstance(cand, str):
            return "Unknown config"
        g3_idx = cand.find("google3/")
        return cand[g3_idx:] if g3_idx != -1 else cand
