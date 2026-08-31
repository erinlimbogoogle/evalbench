import json
import unittest
from unittest.mock import MagicMock, patch
from evaluator.cortadoorchestrator import CortadoOrchestrator


class TestCortadoOrchestrator(unittest.TestCase):

    @patch("evaluator.cortadoorchestrator.CortadoEvaluator")
    def test_evaluate_and_process_returns_5_tuple(self, mock_evaluator_class):
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = (
            [{"eval_id": "c1", "output": "sample_output"}],
            [{"eval_id": "c1", "score": 100}],
        )
        mock_evaluator_class.return_value = mock_evaluator

        config = {
            "runners": {"eval_runners": 1},
            "model_config": "fake_model_config",
        }
        orchestrator = CortadoOrchestrator(
            config=config,
            db_configs={},
            setup_config={},
        )

        orchestrator.evaluate([MagicMock()])

        # Call process and verify it returns a 5-tuple matching the Orchestrator contract
        result = orchestrator.process()
        self.assertEqual(len(result), 5)

        job_id, run_time, results_tf, scores_tf, multi_trial_scores_tf = result
        self.assertEqual(job_id, orchestrator.job_id)
        self.assertEqual(run_time, orchestrator.run_time)
        self.assertIsNotNone(results_tf)
        self.assertIsNotNone(scores_tf)
        self.assertIsNone(multi_trial_scores_tf)

        # Verify contents dumped to temp files
        with open(results_tf, "r") as f:
            results_data = json.load(f)
            self.assertEqual(len(results_data), 1)
            self.assertEqual(results_data[0]["eval_id"], "c1")

        with open(scores_tf, "r") as f:
            scores_data = json.load(f)
            self.assertEqual(len(scores_data), 1)
            self.assertEqual(scores_data[0]["score"], 100)

    def test_extract_golden_sql_structured_turns(self):
        from evaluator.cortadoevaluator import extract_golden_sql_for_turn
        scenario = {
            "turns": [
                {"turn": 1, "golden_sql": "SELECT col1 FROM tbl1;"},
                {"turn": 2, "golden_sql": ["SELECT col2 FROM tbl2;"]},
            ]
        }
        self.assertEqual(
            extract_golden_sql_for_turn(scenario, 0),
            "SELECT col1 FROM tbl1;",
        )
        self.assertEqual(
            extract_golden_sql_for_turn(scenario, 1),
            "SELECT col2 FROM tbl2;",
        )

    def test_extract_golden_sql_conversation_plan(self):
        from evaluator.cortadoevaluator import extract_golden_sql_for_turn
        scenario = {
            "conversation_plan": (
                "Turn 1: The user asks for spending. Agent should execute SQL: 'SELECT id, sum(spending) FROM schools;'\n"
                "Turn 2: The user filters by district. Agent should execute SQL: 'SELECT id, sum(spending) FROM schools WHERE district = 12;'"
            )
        }
        self.assertEqual(
            extract_golden_sql_for_turn(scenario, 0),
            "SELECT id, sum(spending) FROM schools;",
        )
        self.assertEqual(
            extract_golden_sql_for_turn(scenario, 1),
            "SELECT id, sum(spending) FROM schools WHERE district = 12;",
        )

    def test_extract_tools_and_skills_telemetry(self):
        from evaluator.cortadoevaluator import extract_tools_and_skills_from_turn
        eval_result = MagicMock()
        eval_result.other = {
            "macchiato_debug_info": json.dumps({
                "actions": [
                    {"actionName": "dataplex_search"},
                    {"actionName": "query_data_tool"}
                ]
            })
        }
        tools, skills = extract_tools_and_skills_from_turn(
            eval_result,
            agent_text="Here are the results",
            sql_reply="SELECT 1;"
        )
        self.assertIn("dataplex_search", tools)
        self.assertIn("query_data_tool", tools)

    def test_eval_cortado_request_other_proto_roundtrip(self):
        from dataset.cortadoinput import EvalCortadoRequest
        req = EvalCortadoRequest(
            raw_dict={"id": "123", "starting_prompt": "test prompt"}
        )
        req.other = {"macchiato_debug_info": "sample_debug_info"}

        proto = req.to_proto()
        self.assertEqual(proto.other["macchiato_debug_info"], "sample_debug_info")

        restored = EvalCortadoRequest.init_from_proto(proto)
        self.assertEqual(restored.other.get("macchiato_debug_info"), "sample_debug_info")

    @patch("evaluator.cortadoevaluator.databases.get_database")
    @patch("evaluator.cortadoevaluator.GrpcProxyModel")
    def test_cortado_evaluator_multi_turn_execution(self, mock_grpc_model, mock_get_database):
        from evaluator.cortadoevaluator import CortadoEvaluator
        from dataset.cortadoinput import EvalCortadoRequest

        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            ([{"col": 1}], None, None),  # Turn 1 golden
            ([{"col": 1}], None, None),  # Turn 1 generated
            ([{"col": 2}], None, None),  # Turn 2 golden
            ([{"col": 2}], None, None),  # Turn 2 generated
        ]
        mock_get_database.return_value = mock_db

        mock_generator = MagicMock()
        def mock_generate(eval_result):
            if "district" in eval_result.nl_prompt:
                eval_result.generated_nl_response = "Filtered results"
                eval_result.generated_sql = "SELECT col FROM table WHERE district = 1;"
            else:
                eval_result.generated_nl_response = "Initial results"
                eval_result.generated_sql = "SELECT col FROM table;"
        mock_generator.generate.side_effect = mock_generate
        mock_grpc_model.return_value = mock_generator

        config = {
            "model_config": {"generator": "grpc_proxy"},
            "scorers": {"set_match": {}},
            "runners": {"agent_runners": 1}
        }
        evaluator = CortadoEvaluator(config=config, db_configs={"bigquery": [{"db_type": "bigquery"}]})

        scenario = {
            "id": "scenario_01",
            "starting_prompt": "Find all schools",
            "max_turns": 2,
            "conversation_plan": (
                "Turn 1: Find all schools. Agent should execute SQL: 'SELECT col FROM table;'\n"
                "Turn 2: Filter by district. Agent should execute SQL: 'SELECT col FROM table WHERE district = 1;'"
            ),
            "database": "test_db",
            "dialects": ["bigquery"]
        }
        eval_result = EvalCortadoRequest(raw_dict=scenario)

        simulated_user = MagicMock()
        simulated_user.get_next_response.return_value = "Now filter by district 1"

        evaluator.process_scenario(
            scenario=scenario,
            eval_result=eval_result,
            job_id="test_job",
            metadata={"dialects": ["bigquery"], "database": "test_db"},
            simulated_user=simulated_user
        )

        self.assertEqual(len(eval_result.agent_results), 1)
        final_output = eval_result.agent_results[0]
        self.assertIn("turn_history", final_output)
        self.assertEqual(len(final_output["turn_history"]), 2)

        turn_1 = final_output["turn_history"][0]
        self.assertEqual(turn_1["turn"], 1)
        self.assertEqual(turn_1["set_match"], 100.0)
        self.assertEqual(turn_1["generated_sql"], "SELECT col FROM table;")

        turn_2 = final_output["turn_history"][1]
        self.assertEqual(turn_2["turn"], 2)
        self.assertEqual(turn_2["set_match"], 100.0)
        self.assertEqual(turn_2["generated_sql"], "SELECT col FROM table WHERE district = 1;")

    @patch("work.agentscorework.scorer.compare")
    def test_agent_score_work_populates_turn_data(self, mock_scorer_compare):
        from work.agentscorework import AgentScoreWork
        eval_output = {
            "eval_id": "test_eval",
            "scenario": {"starting_prompt": "Test prompt"},
            "metadata": {"dialects": ["bigquery"], "database": "test_db"},
            "turn_history": [
                {"turn": 1, "golden_sql": "SELECT 1;", "generated_sql": "SELECT 1;"},
                {"turn": 2, "golden_sql": "SELECT 2;", "generated_sql": "SELECT 2;"},
            ],
            "generated_sql": "SELECT 2;",
            "golden_sql": "SELECT 2;",
            "generated_result": [{"val": 2}],
            "golden_result": [{"val": 2}],
            "accumulated_tools": ["dataplex_search", "query_data_tool"],
        }
        scoring_results = []
        work = AgentScoreWork(config={}, eval_output=eval_output, scoring_results=scoring_results)
        work.run()

        mock_scorer_compare.assert_called_once()
        call_kwargs = mock_scorer_compare.call_args.kwargs
        scoring_item = call_kwargs["eval_output_item"]

        self.assertEqual(scoring_item["golden_sql"], "SELECT 2;")
        self.assertEqual(scoring_item["generated_sql"], "SELECT 2;")
        self.assertEqual(scoring_item["generated_result"], [{"val": 2}])
        self.assertEqual(scoring_item["accumulated_tools"], ["dataplex_search", "query_data_tool"])

        # Check multi-turn dual rollup metrics and composite score in scoring_results
        comparators = {r["comparator"]: r["score"] for r in scoring_results}
        self.assertIn("set_match_all_turns", comparators)
        self.assertIn("set_match_mean", comparators)
        self.assertIn("set_match_turn_1", comparators)
        self.assertIn("set_match_turn_2", comparators)
        self.assertIn("composite_score", comparators)

    @patch("scorers.llmrater.get_generator")
    def test_llmrater_disambiguation_scoring(self, mock_get_generator):
        from scorers.llmrater import LLMRater
        mock_model = MagicMock()
        mock_model.generate.return_value = "PASS -- The agent correctly asked for clarification."
        mock_get_generator.return_value = mock_model

        rater = LLMRater({"model_config": "fake_model", "prompt_template": "brewmax"}, global_models={})

        # Test 1: Ambiguous prompt with clarification question -> PASS (100%)
        score, response = rater.compare(
            nl_prompt="Show revenue for Springfield",
            golden_query="",
            query_type="disambiguation",
            golden_execution_result=[],
            golden_eval_result="",
            golden_error="",
            generated_query="Which Springfield do you mean (Illinois, Missouri, or Massachusetts)?",
            generated_execution_result=[],
            generated_eval_result={"agent_text": "Which Springfield do you mean?"},
            generated_error="",
            is_ambiguous=True,
        )
        self.assertEqual(score, 100.0)

        # Test 2: Ambiguous prompt with blind SQL -> FAIL (0%)
        score_fail, response_fail = rater.compare(
            nl_prompt="Show revenue for Springfield",
            golden_query="",
            query_type="disambiguation",
            golden_execution_result=[],
            golden_eval_result="",
            golden_error="",
            generated_query="SELECT sum(revenue) FROM districts WHERE city='Springfield';",
            generated_execution_result=[{"rev": 100}],
            generated_eval_result={"agent_text": "Here is the revenue."},
            generated_error="",
            is_ambiguous=True,
        )
        self.assertEqual(score_fail, 0.0)
        self.assertIn("blind SQL", response_fail)

        # Test 3: Unambiguous prompt with unnecessary clarification question -> FAIL (0%)
        score_unambig_fail, response_unambig_fail = rater.compare(
            nl_prompt="List all orders in 2023",
            golden_query="SELECT * FROM orders WHERE year = 2023;",
            query_type="dql",
            golden_execution_result=[{"order_id": 1}],
            golden_eval_result="",
            golden_error="",
            generated_query="skipped",
            generated_execution_result=[],
            generated_eval_result={"generated_disambiguation_question": "What kind of orders do you mean?"},
            generated_error="",
            generated_disambiguation_question="What kind of orders do you mean?",
            is_ambiguous=False,
        )
        self.assertEqual(score_unambig_fail, 0.0)
        self.assertIn("unnecessary clarification", response_unambig_fail)

    def test_group_dataset_by_conversation(self):
        from evaluator.cortadoevaluator import _group_dataset_by_conversation
        from dataset.cortadoinput import EvalCortadoRequest

        # 3 flat requests sharing conversation_id conv_999
        flat_dataset = [
            EvalCortadoRequest(raw_dict={
                "id": "t2",
                "conversation_id": "conv_999",
                "turn_id": 2,
                "user_prompt": "Filter by Illinois",
                "golden_sql": "SELECT * FROM orders WHERE state='IL';"
            }),
            EvalCortadoRequest(raw_dict={
                "id": "t1",
                "conversation_id": "conv_999",
                "turn_id": 1,
                "user_prompt": "Show all orders",
                "golden_sql": "SELECT * FROM orders;"
            }),
            EvalCortadoRequest(raw_dict={
                "id": "single_turn_01",
                "conversation_id": "conv_independent",
                "turn_id": 1,
                "user_prompt": "Show products",
                "golden_sql": "SELECT * FROM products;"
            }),
        ]

        grouped = _group_dataset_by_conversation(flat_dataset)
        self.assertEqual(len(grouped), 2)  # conv_999 + conv_independent

        conv_999_item = next(it for it in grouped if it.conversation_id == "conv_999")
        turns = conv_999_item.raw_dict.get("turns", [])
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["turn"], 1)
        self.assertEqual(turns[0]["user_prompt"], "Show all orders")
        self.assertEqual(turns[1]["turn"], 2)
        self.assertEqual(turns[1]["user_prompt"], "Filter by Illinois")

    @patch("evaluator.cortadoevaluator.databases.get_database")
    @patch("evaluator.cortadoevaluator.GrpcProxyModel")
    def test_cortado_evaluator_deterministic_static_replay(self, mock_grpc_model, mock_get_database):
        from evaluator.cortadoevaluator import CortadoEvaluator
        from dataset.cortadoinput import EvalCortadoRequest

        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            ([{"count": 50}], None, None),  # Turn 1 golden
            ([{"count": 50}], None, None),  # Turn 1 generated
            ([{"avg_age": 28.5}], None, None),  # Turn 2 golden
            ([{"avg_age": 28.5}], None, None),  # Turn 2 generated
        ]
        mock_get_database.return_value = mock_db

        mock_generator = MagicMock()
        prompts_seen = []
        def mock_generate(eval_result):
            prompts_seen.append(eval_result.nl_prompt)
            if "female" in eval_result.nl_prompt:
                eval_result.generated_nl_response = "Filtered for females"
                eval_result.generated_sql = "SELECT AVG(age) FROM users WHERE gender = 'F';"
            else:
                eval_result.generated_nl_response = "Acquired in 2023"
                eval_result.generated_sql = "SELECT traffic_source, COUNT(*) FROM users WHERE year=2023 GROUP BY 1;"
        mock_generator.generate.side_effect = mock_generate
        mock_grpc_model.return_value = mock_generator

        config = {
            "model_config": {"generator": "grpc_proxy"},
            "scorers": {"set_match": {}},
            "runners": {"agent_runners": 1},
        }
        evaluator = CortadoEvaluator(config=config, db_configs={"bigquery": [{"db_type": "bigquery"}]})

        # Structured static sequence turns
        scenario = {
            "id": "seq_564",
            "conversation_id": "conv_13075662202254699956",
            "turns": [
                {
                    "turn_id": 1,
                    "user_prompt": "How many users did we acquire in 2023 by source?",
                    "golden_sql": "SELECT traffic_source, COUNT(*) FROM users WHERE year=2023 GROUP BY 1;"
                },
                {
                    "turn_id": 2,
                    "user_prompt": "Can we filter those to just show female users?",
                    "golden_sql": "SELECT AVG(age) FROM users WHERE gender = 'F';"
                }
            ],
            "database": "anarres-bigquery-test",
            "dialects": ["bigquery"]
        }
        eval_result = EvalCortadoRequest(raw_dict=scenario)

        # Simulated user should NOT be called when static turns exist
        mock_simulated_user = MagicMock()

        evaluator.process_scenario(
            scenario=scenario,
            eval_result=eval_result,
            job_id="test_replay_job",
            metadata={"dialects": ["bigquery"], "database": "anarres-bigquery-test"},
            simulated_user=mock_simulated_user
        )

        mock_simulated_user.get_next_response.assert_not_called()
        self.assertEqual(len(prompts_seen), 2)
        self.assertEqual(prompts_seen[0], "How many users did we acquire in 2023 by source?")
        self.assertEqual(prompts_seen[1], "Can we filter those to just show female users?")

        final_output = eval_result.agent_results[0]
        self.assertEqual(final_output["id"], "seq_564")
        self.assertEqual(final_output["eval_id"], "seq_564")
        self.assertEqual(final_output["nl_prompt"], "How many users did we acquire in 2023 by source?")
        self.assertEqual(final_output["prompt"], "How many users did we acquire in 2023 by source?")
        self.assertEqual(len(final_output["turn_history"]), 2)
        self.assertEqual(final_output["turn_history"][0]["set_match"], 100.0)
        self.assertEqual(final_output["turn_history"][1]["set_match"], 100.0)

        # Verify conversation_history JSON formatting
        conv_hist = json.loads(final_output["conversation_history"])
        self.assertEqual(len(conv_hist), 4)  # 2 user turns + 2 assistant turns
        self.assertEqual(conv_hist[0]["role"], "user")
        self.assertEqual(conv_hist[1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
