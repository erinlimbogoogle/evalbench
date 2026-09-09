from evaluator.orchestrator import Orchestrator, sanitize_eval_output, dump_compact_json
import uuid
import datetime
import tempfile
import json
from dataset.cortadoinput import EvalCortadoRequest
from evaluator.cortadoevaluator import CortadoEvaluator


class CortadoOrchestrator(Orchestrator):
    def __init__(self, config, db_configs, setup_config, report_progress=False):
        self.config = config
        self.db_configs = db_configs
        self.setup_config = setup_config
        self.job_id = f"{uuid.uuid4()}"
        self.run_time = datetime.datetime.now()
        self.total_eval_outputs = []
        self.total_scoring_results = []

    def evaluate(self, dataset: list[EvalCortadoRequest]):
        evaluator = CortadoEvaluator(self.config, db_configs=self.db_configs)
        eval_outputs, scoring_results = evaluator.evaluate(
            dataset, self.job_id, self.run_time
        )
        self.total_eval_outputs.extend(eval_outputs)
        self.total_scoring_results.extend(scoring_results)

    def process(self):
        sanitized_evals = [sanitize_eval_output(item) for item in self.total_eval_outputs]
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            dump_compact_json(sanitized_evals, f)
            results_tf = f.name
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            dump_compact_json(self.total_scoring_results, f)
            scores_tf = f.name
        return self.job_id, self.run_time, results_tf, scores_tf, None
