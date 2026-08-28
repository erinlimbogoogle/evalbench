"""Unit tests for evalbench.rescore module."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from rescore import _safe_parse, _row_to_eval_output, load_traces, rescore


class TestRescore(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_safe_parse(self):
        self.assertEqual(_safe_parse(None, default=[]), [])
        self.assertEqual(_safe_parse("null", default={}), {})
        self.assertEqual(_safe_parse('{"a": 1}'), {"a": 1})
        self.assertEqual(_safe_parse("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(_safe_parse("plain string"), "plain string")

    def test_row_to_eval_output(self):
        row = {
            "id": "scenario_1",
            "nl_prompt": "Show all users",
            "generated_sql": "SELECT * FROM users;",
            "golden_sql": "SELECT * FROM users;",
            "generated_result": json.dumps([{"id": 1}]),
            "golden_result": json.dumps([{"id": 1}]),
            "conversation_history": json.dumps([
                {"role": "user", "content": "Show all users"},
                {"role": "assistant", "content": "SELECT * FROM users;"}
            ]),
            "turn_history": json.dumps([
                {"turn": 1, "set_match": 100.0, "user_prompt": "Show all users"}
            ]),
        }
        eval_output = _row_to_eval_output(row)
        self.assertEqual(eval_output["id"], "scenario_1")
        self.assertEqual(eval_output["eval_id"], "scenario_1")
        self.assertEqual(eval_output["nl_prompt"], "Show all users")
        self.assertEqual(eval_output["generated_sql"], "SELECT * FROM users;")
        self.assertIsInstance(eval_output["turn_history"], list)
        self.assertEqual(len(eval_output["turn_history"]), 1)

    @patch("rescore.load_yaml_config")
    @patch("rescore.load_session_configs")
    @patch("rescore.AgentScoreWork")
    def test_rescore_end_to_end(self, mock_score_work_cls, mock_load_session, mock_load_yaml):
        # Setup mock configs
        mock_load_yaml.return_value = {
            "scorers": {"llmrater": {}, "set_match": {}},
            "model_config": {},
        }
        mock_load_session.return_value = (
            {"scorers": {"llmrater": {}, "set_match": {}}},
            {},
            {},
            {},
        )

        def fake_run(self_work):
            self_work.scoring_results.append({
                "id": self_work.eval_output["id"],
                "comparator": "llmrater",
                "score": 100.0,
            })
            self_work.scoring_results.append({
                "id": self_work.eval_output["id"],
                "comparator": "set_match",
                "score": 100.0,
            })

        mock_work_instance = MagicMock()
        mock_work_instance.run.side_effect = lambda: fake_run(mock_work_instance)
        mock_score_work_cls.side_effect = lambda **kwargs: _create_mock_work(kwargs, fake_run)

        # Create dummy evals.csv
        csv_path = os.path.join(self.test_dir, "evals.csv")
        df = pd.DataFrame([
            {
                "id": "s1",
                "nl_prompt": "Test query 1",
                "generated_sql": "SELECT 1;",
                "golden_sql": "SELECT 1;",
                "generated_result": "[{\"col\": 1}]",
                "golden_result": "[{\"col\": 1}]",
            },
            {
                "id": "s2",
                "nl_prompt": "Test query 2",
                "generated_sql": "SELECT 2;",
                "golden_sql": "SELECT 2;",
                "generated_result": "[{\"col\": 2}]",
                "golden_result": "[{\"col\": 2}]",
            }
        ])
        df.to_csv(csv_path, index=False)

        out_dir = os.path.join(self.test_dir, "rescored_output")
        summary_df = rescore(
            results_file=csv_path,
            config_file="dummy_config.yaml",
            output_dir=out_dir,
            workers=2,
        )

        self.assertTrue(os.path.exists(os.path.join(out_dir, "scores.csv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "summary.csv")))
        self.assertFalse(summary_df.empty)


def _create_mock_work(kwargs, fake_run_fn):
    work = MagicMock()
    work.eval_output = kwargs["eval_output"]
    work.scoring_results = kwargs["scoring_results"]
    work.run.side_effect = lambda: fake_run_fn(work)
    return work


if __name__ == "__main__":
    unittest.main()
