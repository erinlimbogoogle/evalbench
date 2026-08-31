"""Unit tests for evalbench.rescore module."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

import threading
from rescore import (
    _safe_parse,
    _row_to_eval_output,
    _clean_sql_query,
    _reexecute_trace_sql,
    load_traces,
    rescore,
)


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

    def test_clean_sql_query(self):
        self.assertEqual(_clean_sql_query("SELECT 1;"), "SELECT 1;")
        self.assertEqual(
            _clean_sql_query("```sql\nSELECT * FROM users;\n```"),
            "SELECT * FROM users;",
        )
        self.assertEqual(
            _clean_sql_query("```\nSELECT * FROM products;\n```"),
            "SELECT * FROM products;",
        )
        self.assertEqual(
            _clean_sql_query("-- [Sherlog Trace] telemetry 123\nSELECT id FROM items;"),
            "SELECT id FROM items;",
        )
        self.assertEqual(_clean_sql_query("skipped"), "")
        self.assertEqual(_clean_sql_query(""), "")
        self.assertEqual(_clean_sql_query(None), "")

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

    @patch("databases.get_database")
    def test_reexecute_trace_sql_multiturn(self, mock_get_db):
        mock_db = MagicMock()
        mock_db.execute.return_value = ([{"col": "val"}], None, None)
        mock_get_db.return_value = mock_db

        eval_output = {
            "id": "scenario_mt",
            "eval_id": "scenario_mt",
            "database": "test_db",
            "generated_sql": "```sql\nSELECT 1;\n```",
            "golden_sql": "SELECT 1;",
            "generated_result": [],
            "generated_error": "Previous error",
            "golden_result": [],
            "turn_history": [
                {
                    "turn": 1,
                    "user_prompt": "Turn 1 prompt",
                    "generated_sql": "```sql\nSELECT turn1;\n```",
                    "golden_sql": "SELECT turn1;",
                    "generated_execution_result": [],
                    "golden_execution_result": [],
                    "set_match": 0.0,
                }
            ],
            "metadata": {},
            "scenario": {},
        }

        mock_set_matcher = MagicMock()
        mock_set_matcher.compare.return_value = (100.0, "")

        db_cache = {}
        db_lock = threading.Lock()
        _reexecute_trace_sql(
            eval_output=eval_output,
            db_configs={},
            db_cache=db_cache,
            db_lock=db_lock,
            set_matcher=mock_set_matcher,
            force=False,
            only_on_empty=True,
        )

        self.assertEqual(eval_output["generated_sql"], "SELECT 1;")
        self.assertEqual(eval_output["generated_result"], [{"col": "val"}])
        self.assertIsNone(eval_output["generated_error"])
        self.assertEqual(eval_output["golden_result"], [{"col": "val"}])

        # Verify turn_history refreshed
        turn_0 = eval_output["turn_history"][0]
        self.assertEqual(turn_0["generated_sql"], "SELECT turn1;")
        self.assertEqual(turn_0["generated_execution_result"], [{"col": "val"}])
        self.assertEqual(turn_0["golden_execution_result"], [{"col": "val"}])
        self.assertEqual(turn_0["set_match"], 100.0)

    @patch("databases.get_database")
    @patch("rescore.load_yaml_config")
    @patch("rescore.load_session_configs")
    @patch("rescore.AgentScoreWork")
    def test_rescore_end_to_end_with_reexecute(
        self, mock_score_work_cls, mock_load_session, mock_load_yaml, mock_get_db
    ):
        # Setup mock DB
        mock_db = MagicMock()
        mock_db.execute.return_value = ([{"count": 42}], None, None)
        mock_get_db.return_value = mock_db

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

        # Create dummy evals.csv with empty generated_result
        csv_path = os.path.join(self.test_dir, "evals.csv")
        df = pd.DataFrame([
            {
                "id": "s1",
                "nl_prompt": "Test query 1",
                "generated_sql": "SELECT 1;",
                "golden_sql": "SELECT 1;",
                "generated_result": "[]",
                "golden_result": "[]",
            }
        ])
        df.to_csv(csv_path, index=False)

        out_dir = os.path.join(self.test_dir, "rescored_output")
        summary_df = rescore(
            results_file=csv_path,
            config_file="dummy_config.yaml",
            output_dir=out_dir,
            workers=2,
            reexecute_on_empty=True,
            save_refreshed_evals=True,
        )

        self.assertTrue(os.path.exists(os.path.join(out_dir, "scores.csv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "evals_refreshed.csv")))
        self.assertFalse(summary_df.empty)


def _create_mock_work(kwargs, fake_run_fn):
    work = MagicMock()
    work.eval_output = kwargs["eval_output"]
    work.scoring_results = kwargs["scoring_results"]
    work.run.side_effect = lambda: fake_run_fn(work)
    return work


if __name__ == "__main__":
    unittest.main()
