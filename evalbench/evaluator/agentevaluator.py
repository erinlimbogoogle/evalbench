from typing import Any, List
import datetime
import concurrent.futures
import logging
import os
import shutil
import threading

from dataset.evalgeminicliinput import EvalGeminiCliRequest
from generators.models import get_generator
from generators.models.agent_cli import AgentCliGenerator
from mp import mprunner
from work.agentgenwork import AgentGenWork
from evaluator.simulateduser import SimulatedUser
from work.agentscorework import AgentScoreWork
import json
import subprocess
from typing import Dict


class AgentEvaluator:
    def __init__(
        self,
        config,
    ):
        self.config = config

        model_config_path = config.get("model_config")
        if not isinstance(model_config_path, str):
            raise ValueError(
                "AgentEvaluator requires `model_config` to be a path to a model YAML")

        global_models = {
            "lock": threading.Lock(),
            "registered_models": {},
        }
        self.generator = get_generator(global_models, model_config_path)

        if not isinstance(self.generator, AgentCliGenerator):
            raise ValueError(
                f"AgentEvaluator only supports agent CLI generators "
                f"(gemini_cli, claude_code, codex_cli, agy_cli), got "
                f"{type(self.generator).__name__}")
        self.agent_version = self.generator.version

        runner_config = self.config.get("runners", {})
        self.agent_runners = runner_config.get("agent_runners", 10)
        self.agentrunner = mprunner.MPRunner(self.agent_runners)

    def evaluate(
        self,
        dataset: List[EvalGeminiCliRequest],
        job_id: str,
        run_time: datetime.datetime,
    ):
        if isinstance(self.generator, AgentCliGenerator):
            return self._evaluate_agent_cli(dataset, job_id, run_time)
        else:
            raise NotImplementedError(
                "This evaluator currently only supports GeminiCliGenerator, ClaudeCodeGenerator, CodexCliGenerator and AgyCliGenerator")

    def _evaluate_agent_cli(
        self,
        dataset: List[EvalGeminiCliRequest],
        job_id: str,
        run_time: datetime.datetime,
    ):
        eval_outputs: List[Any] = []
        scoring_results: List[Any] = []
        generator_name = type(self.generator).__name__
        logging.info(f"Running {generator_name} evaluation")

        self.agentrunner.futures.clear()

        # Extract generic metadata
        metadata = {
            "dialects": self.config.get("dialects", []),
            "database": self.config.get("database", "unknown"),
            "scorers": self.config.get("scorers", {}),
        }

        for item in dataset:
            simulated_user = SimulatedUser(self.config)
            work = AgentGenWork(
                processor=self.process_scenario,
                eval_result=item,
                job_id=job_id,
                metadata=metadata,
                simulated_user=simulated_user
            )
            self.agentrunner.execute_work(work)

        for future in concurrent.futures.as_completed(self.agentrunner.futures):
            item = future.result()

            if hasattr(item, "agent_results"):
                eval_outputs.extend(item.agent_results)
            if hasattr(item, "scoring_results"):
                scoring_results.extend(item.scoring_results)

        return eval_outputs, scoring_results

    def process_scenario(
        self,
        scenario: Dict[str, Any],
        eval_result: Any,
        job_id: str,
        metadata: Dict[str, Any],
        simulated_user: Any = None
    ):
        """Processes a single scenario."""
        turns = scenario.get("turns", [])
        is_static_replay = bool(turns) or self.config.get("static_replay", False)
        if turns:
            max_turns = len(turns)
        else:
            max_turns = scenario.get("max_turns", 1)

        current_prompt = ""
        if turns and len(turns) > 0:
            current_prompt = turns[0].get("user_prompt") or turns[0].get("prompt") or scenario.get("starting_prompt", "")
        else:
            current_prompt = scenario.get("starting_prompt", "")

        env = scenario.get("env", {})
        conversation_plan = scenario.get("conversation_plan", "")
        conversation_history = []
        accumulated_tools = []
        accumulated_skills = []
        last_result = None

        resolved_work_dir = scenario.get("resolved_work_dir")
        if resolved_work_dir:
            os.makedirs(resolved_work_dir, exist_ok=True)

        # Copy declared env_files to fake_home
        fake_home = getattr(self.generator, "fake_home", None)
        if fake_home:
            session_dir = os.path.dirname(fake_home)
            env_files = scenario.get("env_files", [])
            for env_file in env_files:
                src_path = os.path.join(session_dir, "env_files", env_file)
                dest_path = os.path.join(fake_home, env_file)
                if os.path.exists(src_path):
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(src_path, dest_path)
                    logging.info(
                        "Natively copied env file to sandbox: %s", dest_path
                    )
                else:
                    logging.warning(
                        "Declared env file not found in session: %s", src_path
                    )

        session_id = None
        for turn in range(max_turns):
            if is_static_replay and turns and turn < len(turns):
                current_prompt = turns[turn].get("user_prompt") or turns[turn].get("prompt") or current_prompt

            logging.info(
                f"Turn {turn + 1}/{max_turns} - Prompt: {current_prompt}")
            if isinstance(self.generator, AgentCliGenerator):
                cli_cmd = self.generator.create_command(
                    cli=self.agent_version,
                    prompt=current_prompt,
                    env=env,
                    resume=(turn > 0),
                    session_id=session_id,
                    cwd=resolved_work_dir,
                )
                try:
                    result = self.generator.safe_generate(cli_cmd)
                    if result.stdout:
                        parsed = self.generator.parse_response(result.stdout)
                        if parsed.get("session_id"):
                            session_id = parsed["session_id"]
                except Exception as e:
                    logging.error(f'CLI execution failed: {e}')
                    result = subprocess.CompletedProcess(
                        args=[self.agent_version], returncode=1, stdout='', stderr=str(e)
                    )
            else:
                try:
                    result = self.generator.generate(current_prompt)
                except Exception as e:
                    logging.error(f'LLM generation failed: {e}')
                    result = str(e)

            last_result = result

            self._log_cli_result(turn, max_turns, result)

            tools = []
            if isinstance(self.generator, AgentCliGenerator):
                tools = self.generator.extract_tools(result.stdout)
            accumulated_tools.extend(tools)

            # Extract skills from generator output
            if isinstance(self.generator, AgentCliGenerator):
                skills = self.generator.extract_skills(result.stdout)
                accumulated_skills.extend(skills)

            conversation_history.append({
                "user": current_prompt,
                "agent": result.stdout
            })

            # Next turn determination: static replay vs simulated user
            if turn < max_turns - 1:
                if is_static_replay and turns and (turn + 1) < len(turns):
                    current_prompt = turns[turn + 1].get("user_prompt") or turns[turn + 1].get("prompt") or ""
                elif simulated_user:
                    next_response = simulated_user.get_next_response(
                        conversation_plan,
                        conversation_history,
                        result.stdout
                    )
                    if "TERMINATE" in next_response:
                        logging.info("Simulated user terminated conversation.")
                        break
                    current_prompt = next_response
                else:
                    break

        if last_result:
            self._finalize_scenario(
                scenario,
                last_result,
                conversation_history,
                accumulated_tools,
                accumulated_skills,
                eval_result,
                job_id,
                metadata
            )

    def _log_cli_result(self, turn: int, max_turns: int, result: subprocess.CompletedProcess):
        generator_name = self.generator.name
        logging.info(
            f"Turn {turn + 1}/{max_turns} - {generator_name} exit code: {result.returncode}")
        logging.info(
            f"Turn {turn + 1}/{max_turns} - {generator_name} stdout: {result.stdout}")
        logging.info(
            f"Turn {turn + 1}/{max_turns} - {generator_name} stderr: {result.stderr}")

    def _finalize_scenario(
        self,
        scenario: Dict[str, Any],
        last_result: subprocess.CompletedProcess,
        conversation_history: List[Dict[str, str]],
        accumulated_tools: List[str],
        accumulated_skills: List[str],
        eval_result: Any,
        job_id: str,
        metadata: Dict[str, Any]
    ):
        """Finalizes the scenario by scoring and appending results."""
        # Standardize conversation_history into role/content structure
        formatted_history = []
        for item in conversation_history:
            if isinstance(item, dict):
                if "role" in item and "content" in item:
                    formatted_history.append(item)
                else:
                    if "user" in item:
                        formatted_history.append({"role": "user", "content": str(item["user"])})
                    if "agent" in item:
                        formatted_history.append({"role": "assistant", "content": str(item["agent"])})

        if not formatted_history:
            formatted_history = [
                {"role": "user", "content": scenario.get("starting_prompt", "")},
                {"role": "assistant", "content": last_result.stdout if hasattr(last_result, "stdout") else str(last_result)}
            ]

        # Extract prompt with fallbacks
        prompt_text = (
            scenario.get("starting_prompt")
            or scenario.get("nl_prompt")
            or (
                scenario.get("turns", [{}])[0].get("user_prompt")
                if scenario.get("turns")
                else ""
            )
            or ""
        )

        # Extract scenario ID with fallbacks
        scenario_id = (
            scenario.get("id")
            or scenario.get("eval_id")
            or scenario.get("conversation_id")
            or scenario.get("user_turn_sequence_id")
            or ""
        )

        # Prepare intermediate eval_output with all necessary data for scoring
        eval_output_data = {
            "id": scenario_id,
            "eval_id": scenario_id,
            "nl_prompt": prompt_text,
            "prompt": prompt_text,
            "stdout": last_result.stdout,
            "stderr": last_result.stderr,
            "returncode": last_result.returncode,
            "prompt_generator_error": None,
            "generated_error": None,
            "sql_generator_error": None,
            "golden_error": None,
            "generated_sql": "skipped",
            "conversation_history": json.dumps(formatted_history, indent=2),
            "scenario": scenario,
            "accumulated_tools": accumulated_tools,
            "accumulated_skills": accumulated_skills,
            "job_id": job_id,
            "metadata": metadata,
            "fake_home": self.generator.fake_home if hasattr(self.generator, "fake_home") else None
        }

        score_work = AgentScoreWork(
            config=self.config,
            eval_output=eval_output_data,
            scoring_results=eval_result.scoring_results
        )
        score_work.run()

        eval_result.agent_results.append(eval_output_data)
