import json
import os
import time
from pathlib import Path
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from overrides import override



class DeepSeekAPIHandler(OpenAICompletionsHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "DeepSeekAPIHandler requires DEEPSEEK_API_KEY (or OPENAI_API_KEY) to be set."
            )
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key

        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        self.client = OpenAI(
            base_url=base,
            api_key=api_key,
        )

    # The deepseek API is unstable at the moment, and will frequently give empty responses, so retry on JSONDecodeError is necessary
    @retry_with_backoff(error_type=[RateLimitError, json.JSONDecodeError])
    def generate_with_backoff(self, **kwargs):
        """
        Per the DeepSeek API documentation:
        https://api-docs.deepseek.com/quick_start/rate_limit

        DeepSeek API does NOT constrain user's rate limit. We will try out best to serve every request.
        But please note that when our servers are under high traffic pressure, you may receive 429 (Rate Limit Reached) or 503 (Server Overloaded). When this happens, please wait for a while and retry.

        Thus, backoff is still useful for handling 429 and 503 errors.
        """
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # Save prompt directly if prompt_log_dir is available
        prompt_log_dir = inference_data.get("prompt_log_dir")
        if prompt_log_dir is not None:
            prompt_log_dir = Path(prompt_log_dir)
            test_case_id = inference_data.get("test_case_id", "unknown")
            turn_idx = inference_data.get("turn_idx", 0)
            step_idx = inference_data.get("step_idx", 0)

            try:
                serializable_messages = self._convert_messages_for_logging(message)

                # Determine if this is training or testing based on directory structure
                # Training: prompt_log_dir is already prompt_log_dir / "training" / entry_id
                # Testing: prompt_log_dir is base, need to construct prompt_log_dir / "testing" / test_case_id
                if "training" in str(prompt_log_dir):
                    # Training: save directly in the provided directory
                    prompt_log_path = prompt_log_dir
                else:
                    # Testing: construct the path
                    prompt_log_path = prompt_log_dir / "testing" / test_case_id

                prompt_log_path.mkdir(parents=True, exist_ok=True)

                # For training, save as generator_prompt.json (single file per entry, overwrites each step)
                # For testing, save as generator_prompt_turn_{turn_idx}_step_{step_idx}.json
                if "training" in str(prompt_log_dir):
                    # For training, save/overwrite the prompt file (typically only one step per entry)
                    prompt_file = prompt_log_path / "generator_prompt.json"
                    prompt_data = {
                        "messages": serializable_messages,
                        "tools": tools if len(tools) > 0 else None,
                    }
                else:
                    prompt_file = prompt_log_path / f"generator_prompt_turn_{turn_idx}_step_{step_idx}.json"
                    prompt_data = {
                        "messages": serializable_messages,
                        "tools": tools if len(tools) > 0 else None,
                    }

                with open(prompt_file, "w", encoding="utf-8") as f:
                    json.dump(prompt_data, f, indent=2, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                print(
                    f"[Warning] Failed to save generator prompt for {test_case_id} "
                    f"(turn {turn_idx}, step {step_idx}): {exc}"
                )

        if len(tools) > 0:
            return self.generate_with_backoff(
                model=self.model_name,
                messages=message,
                tools=tools,
                temperature=self.temperature,
            )
        else:
            return self.generate_with_backoff(
                model=self.model_name,
                messages=message,
                temperature=self.temperature,
            )

    @override
    def _query_prompting(self, inference_data: dict):
        """
        This method is intended to be used by the `DeepSeek-R1` models. If used for other models, you will need to modify the code accordingly.

        Reasoning models don't support temperature parameter
        https://api-docs.deepseek.com/guides/reasoning_model

        `DeepSeek-R1` should use `deepseek-reasoner` as the model name in the API
        https://api-docs.deepseek.com/quick_start/pricing
        """
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        # Save prompt directly if prompt_log_dir is available
        prompt_log_dir = inference_data.get("prompt_log_dir")
        if prompt_log_dir is not None:
            prompt_log_dir = Path(prompt_log_dir)
            test_case_id = inference_data.get("test_case_id", "unknown")
            turn_idx = inference_data.get("turn_idx", 0)
            step_idx = inference_data.get("step_idx", 0)

            try:
                serializable_messages = self._convert_messages_for_logging(message)

                # Determine if this is training or testing based on directory structure
                if "training" in str(prompt_log_dir):
                    prompt_log_path = prompt_log_dir
                else:
                    prompt_log_path = prompt_log_dir / "testing" / test_case_id

                prompt_log_path.mkdir(parents=True, exist_ok=True)

                if "training" in str(prompt_log_dir):
                    prompt_file = prompt_log_path / "generator_prompt.json"
                    prompt_data = {
                        "messages": serializable_messages,
                    }
                else:
                    prompt_file = prompt_log_path / f"generator_prompt_turn_{turn_idx}_step_{step_idx}.json"
                    prompt_data = {
                        "messages": serializable_messages,
                    }

                with open(prompt_file, "w", encoding="utf-8") as f:
                    json.dump(prompt_data, f, indent=2, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                print(
                    f"[Warning] Failed to save generator prompt for {test_case_id} "
                    f"(turn {turn_idx}, step {step_idx}): {exc}"
                )

        return self.generate_with_backoff(
            model=self.model_name,
            messages=message,
        )

    def _convert_messages_for_logging(self, messages):
        serializable = []
        for msg in messages or []:
            if hasattr(msg, "model_dump"):
                payload = msg.model_dump()
            elif hasattr(msg, "dict"):
                payload = msg.dict()
            elif isinstance(msg, dict):
                payload = msg
            else:
                payload = {
                    "role": getattr(msg, "role", "unknown"),
                    "content": getattr(msg, "content", ""),
                }

            try:
                json.dumps(payload, ensure_ascii=False)
                serializable.append(payload)
            except (TypeError, ValueError):
                serializable.append(json.loads(json.dumps(payload, default=str)))

        return serializable

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        # 'deepseek-reasoner does not support successive user messages, so we need to combine them
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available_prompting(api_response, response_data)
        return response_data

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data
