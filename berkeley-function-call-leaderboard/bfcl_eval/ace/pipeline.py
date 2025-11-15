from __future__ import annotations

import json
import random
import shutil
import tempfile
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from tqdm import tqdm

from bfcl_eval._llm_response_generation import (
    build_handler,
    multi_threaded_inference,
)
from bfcl_eval.ace.metrics import PlaybookTrainingMetricsCollector
from bfcl_eval.ace.playbook import PlaybookManager
from bfcl_eval.ace.split_manager import (
    TRAIN_PARTITION,
    ensure_split_exists,
    get_partition_ids,
)
from bfcl_eval.constants.category_mapping import ALL_CATEGORIES
from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.eval_checker.eval_runner import (
    _evaluate_single_agentic_entry,
    _evaluate_single_ast_entry,
    _evaluate_single_multi_turn_entry,
    _evaluate_single_relevance_entry,
)
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.utils import (
    extract_test_category_from_id,
    is_agentic,
    is_chatable,
    is_executable,
    is_java,
    is_js,
    is_memory_prereq,
    is_multi_turn,
    is_relevance_or_irrelevance,
    is_sql,
    load_dataset_entry,
    load_ground_truth_entry,
    populate_initial_settings_for_memory_test_cases,
    populate_initial_settings_for_web_search_test_cases,
)

def _make_json_serializable(obj):
    """Recursively convert sets and other non-serializable types to JSON-serializable types."""
    if isinstance(obj, set):
        return sorted(list(obj))  # Convert set to sorted list for reproducibility
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__class__"):
        # Handle Directory objects (have name, parent, contents)
        if hasattr(obj, "name") and hasattr(obj, "contents") and hasattr(obj, "parent"):
            return {
                "name": obj.name,
                "parent": obj.parent.name if obj.parent else None,
                "contents": _make_json_serializable(obj.contents),
            }
        # Handle File objects (have name, content)
        elif hasattr(obj, "name") and hasattr(obj, "content") and not hasattr(obj, "contents"):
            file_dict = {
                "name": obj.name,
                "content": obj.content,
            }
            # Include _last_modified if present
            if hasattr(obj, "_last_modified"):
                file_dict["_last_modified"] = _make_json_serializable(obj._last_modified)
            return file_dict
        # For other objects, try to convert to dict using vars(), or fallback to string
        else:
            try:
                # Try to serialize directly first
                json.dumps(obj, ensure_ascii=False)
                return obj
            except (TypeError, ValueError):
                # If it fails, try to convert using vars() or __dict__
                try:
                    if hasattr(obj, "__dict__"):
                        return _make_json_serializable(vars(obj))
                    else:
                        return str(obj)
                except Exception:
                    return str(obj)
    else:
        return obj

from .constants import (
    DEFAULT_PLAYBOOK_PATH,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
)
from .llm import ChatCompletionClient
from .utils import (
    camel_to_snake,
    count_available_tools,
    determine_tool_groups,
    extract_base_prompt_id,
    extract_memory_ground_truth_id,
    format_conversation,
    serialize_json,
)


def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove leading fences like ```json\n ... ``` 
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


@dataclass
class DatasetCache:
    prompt: Dict[str, dict]
    ground_truth: Dict[str, dict]
    ground_truth_aliases: Dict[str, dict]

    @classmethod
    def build(cls) -> "DatasetCache":
        prompt: Dict[str, dict] = {}
        ground_truth: Dict[str, dict] = {}
        ground_truth_aliases: Dict[str, dict] = {}

        for category in ALL_CATEGORIES:
            prompts = load_dataset_entry(
                category,
                include_prereq=False,
                include_language_specific_hint=True,
            )
            try:
                ground_truth_entries = load_ground_truth_entry(category)
            except FileNotFoundError:
                ground_truth_entries = []
            gt_map = {entry.get("id"): entry for entry in ground_truth_entries if "id" in entry}
            for entry in prompts:
                entry_id = entry["id"]
                prompt[entry_id] = entry
                if entry_id in gt_map:
                    ground_truth[entry_id] = gt_map[entry_id]
                    continue

                base_id = extract_base_prompt_id(entry_id)
                if base_id in gt_map:
                    ground_truth_aliases[entry_id] = gt_map[base_id]
                    continue

                memory_id = extract_memory_ground_truth_id(entry_id)
                if memory_id and memory_id in gt_map:
                    ground_truth_aliases[entry_id] = gt_map[memory_id]

        return cls(
            prompt=prompt,
            ground_truth=ground_truth,
            ground_truth_aliases=ground_truth_aliases,
        )

    def get_prompt(self, entry_id: str) -> Optional[dict]:
        return self.prompt.get(entry_id)

    def get_ground_truth(self, entry_id: str) -> Optional[dict]:
        if entry_id in self.ground_truth:
            return self.ground_truth[entry_id]
        if entry_id in self.ground_truth_aliases:
            return self.ground_truth_aliases[entry_id]
        return None


def _sanitize_evaluation_details(
    data,
    *,
    max_depth: int = 3,
    list_limit: int = 5,
    string_limit: int = 400,
):
    if max_depth < 0:
        return "...(truncated)"

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key in {
                "prompt",
                "possible_answer",
                "model_result",
                "model_result_raw",
                "model_result_decoded",
                "inference_log",
            }:
                continue
            sanitized[key] = _sanitize_evaluation_details(
                value,
                max_depth=max_depth - 1,
                list_limit=list_limit,
                string_limit=string_limit,
            )
        return sanitized

    if isinstance(data, list):
        if not data:
            return []
        trimmed = data[:list_limit]
        sanitized_items = [
            _sanitize_evaluation_details(
                item,
                max_depth=max_depth - 1,
                list_limit=list_limit,
                string_limit=string_limit,
            )
            for item in trimmed
        ]
        if len(data) > list_limit:
            sanitized_items.append(f"...({len(data) - list_limit} more items)")
        return sanitized_items

    if isinstance(data, str):
        text = data.strip()
        if len(text) > string_limit:
            return text[: string_limit - 3] + "..."
        return text

    return data


def _extract_error_info(details: dict) -> tuple[Optional[str], Optional[str]]:
    error_type = details.get("error_type")
    error_message = details.get("error_message")

    error_payload = details.get("error")
    if isinstance(error_payload, dict):
        error_type = error_type or error_payload.get("error_type")
        error_message = error_message or error_payload.get("error_message")
    elif isinstance(error_payload, list) and error_message is None:
        error_message = error_payload

    if isinstance(error_message, list):
        error_message = ", ".join(str(item) for item in error_message)

    return error_type, error_message


def _evaluate_generator_result(
    handler,
    entry: dict,
    generator_result: dict,
    ground_truth: Optional[dict],
) -> dict:
    test_category = extract_test_category_from_id(entry["id"])
    evaluation: dict = {
        "entry_id": entry["id"],
        "test_category": test_category,
        "status": "pending",
    }

    if is_chatable(test_category) or is_sql(test_category) or is_executable(test_category):
        evaluation.update(
            status="skipped",
            reason="BFCL benchmark skips evaluation for this category.",
        )
        return evaluation

    if is_memory_prereq(test_category):
        evaluation.update(
            status="skipped",
            reason="Prerequisite memory categories are not scored in BFCL benchmark.",
        )
        return evaluation

    if "result" not in generator_result:
        evaluation.update(
            status="error",
            error="Generator result does not contain 'result' payload.",
        )
        return evaluation

    if ground_truth is None and not is_relevance_or_irrelevance(test_category):
        evaluation.update(
            status="skipped",
            reason="Ground truth unavailable; benchmark would not score this entry.",
        )
        return evaluation

    result_payload = generator_result.get("result")
    model_name = getattr(handler, "registry_name", getattr(handler, "model_name", "unknown"))

    try:
        if is_multi_turn(test_category):
            entry_result = _evaluate_single_multi_turn_entry(
                handler,
                entry["id"],
                result_payload,
                ground_truth.get("ground_truth") if ground_truth else None,
                deepcopy(entry),
                model_name,
                test_category,
            )
        elif is_agentic(test_category):
            entry_result = _evaluate_single_agentic_entry(
                handler,
                entry["id"],
                result_payload,
                ground_truth.get("ground_truth") if ground_truth else None,
                deepcopy(entry),
                model_name,
                test_category,
            )
        elif is_relevance_or_irrelevance(test_category):
            entry_result = _evaluate_single_relevance_entry(
                handler,
                entry["id"],
                result_payload,
                deepcopy(entry),
                model_name,
                test_category,
            )
        else:
            if ground_truth is None:
                entry_result = {
                    "valid": False,
                    "error_type": "ace:evaluation_error",
                    "error_message": "Missing ground truth for AST evaluation.",
                }
            else:
                if is_java(test_category):
                    language = Language.JAVA
                    return_format = ReturnFormat.JAVA
                elif is_js(test_category):
                    language = Language.JAVASCRIPT
                    return_format = ReturnFormat.JAVASCRIPT
                else:
                    language = Language.PYTHON
                    return_format = ReturnFormat.PYTHON

                entry_result = _evaluate_single_ast_entry(
                    handler,
                    entry["id"],
                    result_payload,
                    ground_truth.get("ground_truth"),
                    deepcopy(entry),
                    model_name,
                    test_category,
                    language=language,
                    return_format=return_format,
                    has_tool_call_tag=False,
                )
    except Exception as exc:  # pragma: no cover - defensive catch
        evaluation.update(
            status="error",
            error=f"Exception during evaluation: {exc}",
            traceback=traceback.format_exc(limit=5),
        )
        return evaluation

    evaluation.update(
        status="evaluated",
        valid=entry_result.get("valid"),
        details=entry_result,
        sanitized_details=_sanitize_evaluation_details(entry_result),
    )
    return evaluation


def _format_evaluation_summary(evaluation: dict) -> str:
    status = evaluation.get("status")
    entry_id = evaluation.get("entry_id")
    test_category = evaluation.get("test_category")

    if status == "skipped":
        return (
            f"BFCL evaluation skipped for {entry_id} ({test_category}): "
            f"{evaluation.get('reason', 'no reason provided')}."
        )
    if status == "error":
        return (
            f"BFCL evaluation errored for {entry_id} ({test_category}): "
            f"{evaluation.get('error', 'unexpected error')}."
        )
    if status != "evaluated":
        return f"BFCL evaluation status unknown for {entry_id} ({test_category})."

    details = evaluation.get("details", {})
    if details.get("valid"):
        return f"BFCL evaluation PASSED for {entry_id} ({test_category})."

    error_type, error_message = _extract_error_info(details)
    lines = [
        f"BFCL evaluation FAILED for {entry_id} ({test_category}).",
    ]
    if error_type:
        lines.append(f"Error type: {error_type}")
    if error_message:
        lines.append(f"Error message: {error_message}")
    return "\n".join(lines)


_TOOL_CALL_META_KEYS: Set[str] = {
    "role",
    "content",
    "handler_log",
    "model_response_decoded",
    "model_response_raw",
    "state_info",
    "error",
    "details",
    "inference_log",
    "prompt",
    "status",
    "reason",
    "traceback",
}


def _extract_model_tool_calls_for_turn(obj) -> Set[str]:
    calls: Set[str] = set()
    if isinstance(obj, list):
        for item in obj:
            calls |= _extract_model_tool_calls_for_turn(item)
    elif isinstance(obj, dict):
        candidate_keys = [key for key in obj.keys() if key not in _TOOL_CALL_META_KEYS]
        if len(candidate_keys) == 1:
            key = candidate_keys[0]
            calls.add(key)
            value = obj[key]
            if isinstance(value, list):
                for item in value:
                    calls |= _extract_model_tool_calls_for_turn(item)
        else:
            for key, value in obj.items():
                if key in _TOOL_CALL_META_KEYS:
                    calls |= _extract_model_tool_calls_for_turn(value)
    elif isinstance(obj, str):
        fn_name = obj.split("(", 1)[0].strip()
        if fn_name:
            calls.add(fn_name)
    return calls


def _extract_model_tool_calls(generator_result: dict) -> List[Set[str]]:
    payload = generator_result.get("result")
    if isinstance(payload, list):
        return [_extract_model_tool_calls_for_turn(turn) for turn in payload]
    if payload is None:
        return []
    return [_extract_model_tool_calls_for_turn(payload)]


def _extract_ground_truth_tool_calls_for_turn(obj) -> Set[str]:
    calls: Set[str] = set()
    if isinstance(obj, list):
        for item in obj:
            calls |= _extract_ground_truth_tool_calls_for_turn(item)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            calls.add(key)
            if isinstance(value, list):
                calls |= _extract_ground_truth_tool_calls_for_turn(value)
    elif isinstance(obj, str):
        fn_name = obj.split("(", 1)[0].strip()
        if fn_name:
            calls.add(fn_name)
    return calls


def _extract_ground_truth_tool_calls(ground_truth: Optional[dict]) -> List[Set[str]]:
    if ground_truth is None:
        return []
    payload = ground_truth.get("ground_truth") if isinstance(ground_truth, dict) else ground_truth
    if isinstance(payload, list):
        return [_extract_ground_truth_tool_calls_for_turn(turn) for turn in payload]
    if payload is None:
        return []
    return [_extract_ground_truth_tool_calls_for_turn(payload)]


def _summarize_tool_call_diff(generator_result: dict, ground_truth: Optional[dict]) -> str:
    model_turns = _extract_model_tool_calls(generator_result)
    ground_truth_turns = _extract_ground_truth_tool_calls(ground_truth)
    if not ground_truth_turns:
        return ""

    max_turns = max(len(model_turns), len(ground_truth_turns))
    lines: List[str] = []
    for idx in range(max_turns):
        model_calls = model_turns[idx] if idx < len(model_turns) else set()
        ground_truth_calls = ground_truth_turns[idx] if idx < len(ground_truth_turns) else set()
        missing = sorted(ground_truth_calls - model_calls)
        extra = sorted(model_calls - ground_truth_calls)
        if missing or extra:
            parts: List[str] = []
            if missing:
                parts.append(f"missing {', '.join(missing)}")
            if extra:
                parts.append(f"extra {', '.join(extra)}")
            lines.append(f"Turn {idx}: " + "; ".join(parts))
    return "\n".join(lines)


def _append_tool_call_diff(
    evaluation_summary: str,
    generator_result: dict,
    ground_truth: Optional[dict],
) -> str:
    diff = _summarize_tool_call_diff(generator_result, ground_truth)
    if diff:
        return f"{evaluation_summary}\n\nTool-call differences:\n{diff}"
    return evaluation_summary


def _serialize_evaluation_payload(evaluation: dict) -> str:
    status = evaluation.get("status")
    if status == "evaluated":
        payload = {
            "status": status,
            "valid": evaluation.get("valid"),
            "details": evaluation.get("sanitized_details"),
        }
    else:
        payload = {
            "status": status,
            "reason": evaluation.get("reason"),
            "error": evaluation.get("error"),
            "traceback": evaluation.get("traceback"),
        }
    payload["entry_id"] = evaluation.get("entry_id")
    payload["test_category"] = evaluation.get("test_category")
    return serialize_json({k: v for k, v in payload.items() if v is not None})


def _summarize_generator_output(result: dict) -> str:
    payload = {
        "id": result["id"],
        "result": result["result"],
    }
    metadata_keys = ("reasoning_content", "traceback")
    for key in metadata_keys:
        if key in result:
            payload[key] = result[key]
    return serialize_json(payload)


def _prepare_entry_for_training(entry: dict, model_result_dir: Path) -> dict:
    """
    Prepare a dataset entry for generator inference during playbook training.

    Mirrors the setup performed in the standalone generation pipeline so that
    simulator-backed categories (memory, web search) receive the expected bootstrapping.
    """

    prepared_entry = deepcopy(entry)
    working_list = [prepared_entry]

    populate_initial_settings_for_memory_test_cases(working_list, model_result_dir)
    populate_initial_settings_for_web_search_test_cases(working_list)

    return working_list[0]


def _build_reflector_messages(
    playbook_text: str,
    entry: dict,
    generator_result: dict,
    ground_truth: dict,
    tool_groups: Sequence[str],
    evaluation_summary: str,
    evaluation_details_json: str,
    evaluation_valid: Optional[bool],
) -> List[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "You are the ACE Reflector. Identify precise, short insights about why a "
            "model's tool usage differed from the BFCL ground truth tool calls. Use the evaluation summary "
            "and outcome to focus on genuine divergences. When the evaluation passed, highlight the "
            "key behaviour that matched expectations or note that no corrective insight is needed. "
            "BFCL evaluates whether tool calls, parameters, and state transitions are correct; the "
            "model does not need to mirror the ground-truth text verbatim. "
            "Respond in compact JSON."
        ),
    }
    user_content = f"""
Sample ID: {entry['id']}
Tool groups: {', '.join(tool_groups)}

Evaluation summary:
{evaluation_summary}

Evaluation details (JSON):
{evaluation_details_json}

Outcome:
{"PASSED" if evaluation_valid else "FAILED" if evaluation_valid is not None else "UNKNOWN"}

Current playbook state:
{playbook_text}

Conversation:
{format_conversation(entry['question'])}

Model output:
{_summarize_generator_output(generator_result)}

Ground truth reference:
BFCL evaluates whether the tool interactions and resulting state match the ground truth; verbatim textual matches are not required.

Guidance:
- Anchor your reasoning to the evaluation summary of the tool calls.
- If the issue appears scenario-specific, make that clear instead of implying a universal rule.

Return JSON using this schema (keep each field to <=2 sentences):
{{
  "reasoning": "...",
  "error_identification": "...",
  "root_cause_analysis": "...",
  "correct_approach": "...",
  "key_insight": "..."
}}
"""
    return [system_msg, {"role": "user", "content": user_content.strip()}]


def _build_curator_messages(
    playbook_text: str,
    entry: dict,
    generator_result: dict,
    reflection: dict,
    ground_truth: dict,
    tool_groups: Sequence[str],
    evaluation_summary: str,
    evaluation_details_json: str,
    evaluation_valid: Optional[bool],
) -> List[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "You are the ACE Curator. Convert reflections into crisp playbook operations "
            "per tool group. Prefer modifying existing bullets when possible, but keep different concepts separate. Only generalize "
            "when the evaluation summary indicates a systemic issue, but otherwise don't overgeneralize bullet points. Generally respect the BFCL evaluation but think about what you would want to tell the model in order for it to improve in the many cases it may face in the future"
            ".keep the playbook unchanged on successes unless an existing entry now "
            "conflicts with verified behaviour. Functional parity with the checker matters more than superficial differences "
            ". Output JSON."
        ),
    }
    user_content = f"""
Sample ID: {entry['id']}
Tool groups: {', '.join(tool_groups)}

Existing playbook:
{playbook_text}

Evaluation summary:
{evaluation_summary}

Evaluation details (JSON):
{evaluation_details_json}

Outcome:
{"PASSED" if evaluation_valid else "FAILED" if evaluation_valid is not None else "UNKNOWN"}

Model output:
{_summarize_generator_output(generator_result)}

Ground truth reference:
Focus on achieving the checker-validated tool effects.

Reflection:
{serialize_json(reflection)}

Guidance:
- If the evaluation PASSED, many cases you would return an empty operation list. Propose removals or updates when the current playbook conflicts with the verified behaviour as to need to be updated.
- If the evaluation FAILED, add, modify, or remove bullets narrowly targeted to the failure described in the summary/details.
- Keep updates scoped to the triggering context unless the pattern is clearly systemic.
- Keep content under 160 characters. Return an empty list when no update is needed.

Return JSON with keys "reasoning" (<=2 sentences) and "operations".
Each operation must be one of:
  {{"type": "ADD", "section": "<tool_group>", "content": "<short bullet>"}}
  {{"type": "MODIFY", "section": "<tool_group>", "ID": "<existing id>", "content": "<short bullet>"}}
  {{"type": "REMOVE", "section": "<tool_group>", "ID": "<existing id>"}}
"""
    return [system_msg, {"role": "user", "content": user_content.strip()}]


def train_playbook(
    playbook_path: Path = DEFAULT_PLAYBOOK_PATH,
    split_path: Path = DEFAULT_SPLIT_PATH,
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    generator_model: str = "DeepSeek-V3.2-Exp-FC",
    reflector_model: str = "deepseek-chat",
    curator_model: str = "deepseek-chat",
    generator_temperature: float = 0.001,
    completion_temperature: float = 0.0,
    limit: Optional[int] = None,
    start_offset: int = 0,
    regenerate_split: bool = False,
    num_gpus: int = 1,
    backend: str = "sglang",
    gpu_memory_utilization: float = 0.9,
    skip_server_setup: bool = False,
    local_model_path: Optional[str] = None,
    prompt_log_dir: Optional[Path] = None,
) -> None:
    """
    Run the ACE training pipeline: Generator -> Reflector -> Curator.
    """
    ensure_split_exists(
        seed=split_seed,
        train_ratio=train_ratio,
        output_path=split_path,
        regenerate=regenerate_split,
    )
    train_ids = sorted(get_partition_ids(TRAIN_PARTITION, path=split_path))
    rng = random.Random(split_seed)
    rng.shuffle(train_ids)

    if start_offset:
        if start_offset >= len(train_ids):
            tqdm.write(
                f"[Info] Start offset {start_offset} is beyond dataset size ({len(train_ids)}); nothing to process."
            )
            return
        train_ids = train_ids[start_offset:]

    if limit:
        train_ids = train_ids[:limit]

    planned_sample_count = len(train_ids)
    playbook = PlaybookManager(playbook_path, reset=False)
    dataset_cache = DatasetCache.build()

    handler = build_handler(generator_model, generator_temperature)

    reflector_client = ChatCompletionClient(
        model=reflector_model, temperature=completion_temperature
    )
    curator_client = ChatCompletionClient(
        model=curator_model, temperature=completion_temperature
    )

    metrics_collector = PlaybookTrainingMetricsCollector()
    temp_result_root = Path(tempfile.mkdtemp(prefix="ace_training_"))
    temp_model_result_dir = temp_result_root / getattr(handler, "registry_dir_name", "ace_training_model")
    temp_model_result_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "playbook_path": str(playbook_path),
        "split_path": str(split_path),
        "split_seed": split_seed,
        "train_ratio": train_ratio,
        "generator_model": generator_model,
        "reflector_model": reflector_model,
        "curator_model": curator_model,
        "generator_temperature": generator_temperature,
        "completion_temperature": completion_temperature,
        "limit": limit,
        "start_offset": start_offset,
        "planned_sample_count": planned_sample_count,
    }

    is_oss_model = isinstance(handler, OSSHandler)
    try:
        if is_oss_model:
            handler.spin_up_local_server(
                num_gpus=num_gpus,
                gpu_memory_utilization=gpu_memory_utilization,
                backend=backend,
                skip_server_setup=skip_server_setup,
                local_model_path=local_model_path,
            )

        progress = tqdm(train_ids, desc="ACE training", unit="sample")
        for entry_id in progress:
            entry = dataset_cache.get_prompt(entry_id)
            if not entry:
                metrics_collector.record_sample(
                    entry_id=entry_id,
                    tool_groups=[],
                    applied_operations=[],
                    outcome="missing_entry",
                    evaluation_passed=None,
                    notes="Prompt entry unavailable",
                )
                continue

            ground_truth = dataset_cache.get_ground_truth(entry_id)
            if not ground_truth:
                metrics_collector.record_sample(
                    entry_id=entry_id,
                    tool_groups=[],
                    applied_operations=[],
                    outcome="missing_ground_truth",
                    evaluation_passed=None,
                    notes="Ground truth unavailable",
                )
                continue

            tool_groups = determine_tool_groups(entry)

            # Save generator prompts if logging is enabled
            generator_prompt_log_path = None
            if prompt_log_dir:
                generator_prompt_log_path = (
                    prompt_log_dir / "training" / entry_id
                )
                generator_prompt_log_path.mkdir(parents=True, exist_ok=True)

            prepared_entry = _prepare_entry_for_training(entry, temp_model_result_dir)

            generator_result = multi_threaded_inference(
                handler=handler,
                test_case=prepared_entry,
                include_input_log=False,
                exclude_state_log=False,
                playbook_text=None,
                prompt_log_dir=generator_prompt_log_path,
            )
            # Generator prompts are now saved directly in the handler (deepseek.py)
            # No need to extract from inference_log anymore

            focus_sections = list(dict.fromkeys(tool_groups))
            playbook_text = playbook.to_prompt_string(
                focus_sections=focus_sections or None,
                max_sections=len(focus_sections) if focus_sections else None,
            )
            evaluation = _evaluate_generator_result(
                handler, prepared_entry, generator_result, ground_truth
            )
            evaluation_summary = _format_evaluation_summary(evaluation)
            evaluation_summary = _append_tool_call_diff(
                evaluation_summary,
                generator_result,
                ground_truth,
            )
            evaluation_details_json = _serialize_evaluation_payload(evaluation)
            evaluation_valid = (
                evaluation.get("valid")
                if evaluation.get("status") == "evaluated"
                else None
            )

            reflection_messages = _build_reflector_messages(
                playbook_text,
                prepared_entry,
                generator_result,
                ground_truth,
                tool_groups,
                evaluation_summary,
                evaluation_details_json,
                evaluation_valid,
            )

            # Save reflector prompts if logging is enabled
            if prompt_log_dir:
                reflector_prompt_file = (
                    prompt_log_dir / "training" / entry_id / "reflector_prompt.json"
                )
                reflector_prompt_file.parent.mkdir(parents=True, exist_ok=True)
                # Save exact format that API receives
                reflector_prompt_data = {
                    "messages": reflection_messages,
                }
                with open(reflector_prompt_file, "w", encoding="utf-8") as f:
                    json.dump(reflector_prompt_data, f, indent=2, ensure_ascii=False)

            try:
                reflection_raw = reflector_client.complete(
                    reflection_messages, temperature=completion_temperature
                )
            except Exception as exc:
                msg = f"[Warning] Reflector call failed for {entry_id}: {exc}"
                tqdm.write(msg)
                metrics_collector.record_sample(
                    entry_id=entry_id,
                    tool_groups=tool_groups,
                    applied_operations=[],
                    outcome="reflector_error",
                    evaluation_passed=evaluation_valid,
                    notes=msg,
                )
                continue

            reflection_raw_clean = _extract_json_block(reflection_raw)
            try:
                reflection_json = json.loads(reflection_raw_clean)
            except json.JSONDecodeError as exc:
                msg = f"[Warning] Failed to parse reflector output for {entry_id}: {exc}"
                tqdm.write(msg)
                metrics_collector.record_sample(
                    entry_id=entry_id,
                    tool_groups=tool_groups,
                    applied_operations=[],
                    outcome="reflector_parse_error",
                    evaluation_passed=evaluation_valid,
                    notes=msg,
                )
                continue

            curator_messages = _build_curator_messages(
                playbook_text,
                prepared_entry,
                generator_result,
                reflection_json,
                ground_truth,
                tool_groups,
                evaluation_summary,
                evaluation_details_json,
                evaluation_valid,
            )

            # Save curator prompts if logging is enabled
            if prompt_log_dir:
                curator_prompt_file = (
                    prompt_log_dir / "training" / entry_id / "curator_prompt.json"
                )
                curator_prompt_file.parent.mkdir(parents=True, exist_ok=True)
                # Save exact format that API receives
                curator_prompt_data = {
                    "messages": curator_messages,
                }
                with open(curator_prompt_file, "w", encoding="utf-8") as f:
                    json.dump(curator_prompt_data, f, indent=2, ensure_ascii=False)

            try:
                curator_raw = curator_client.complete(
                    curator_messages, temperature=completion_temperature
                )
            except Exception as exc:
                msg = f"[Warning] Curator call failed for {entry_id}: {exc}"
                tqdm.write(msg)
                metrics_collector.record_sample(
                    entry_id=entry_id,
                    tool_groups=tool_groups,
                    applied_operations=[],
                    outcome="curator_error",
                    evaluation_passed=evaluation_valid,
                    notes=msg,
                )
                continue

            curator_raw_clean = _extract_json_block(curator_raw)
            try:
                curator_json = json.loads(curator_raw_clean)
            except json.JSONDecodeError as exc:
                msg = f"[Warning] Failed to parse curator output for {entry_id}: {exc}"
                tqdm.write(msg)
                metrics_collector.record_sample(
                    entry_id=entry_id,
                    tool_groups=tool_groups,
                    applied_operations=[],
                    outcome="curator_parse_error",
                    evaluation_passed=evaluation_valid,
                    notes=msg,
                )
                continue

            operations = curator_json.get("operations", [])
            applied_any = False
            applied_operations: List[dict] = []
            notes_messages: List[str] = []

            for operation in operations:
                op_type = operation.get("type", "").upper()
                section = operation.get("section")
                if not section:
                    warning = (
                        f"[Warning] Skipping curator operation for {entry_id}: missing 'section' field. "
                        f"Operation type: {op_type}, Available tool groups: {', '.join(tool_groups) if tool_groups else 'none'}"
                    )
                    tqdm.write(warning)
                    notes_messages.append(warning)
                    continue

                normalized_section = camel_to_snake(section)
                if normalized_section not in tool_groups:
                    warning = (
                        f"[Warning] Skipping curator operation for {entry_id}: section '{section}' "
                        f"(normalized: '{normalized_section}') is not one of the tool groups for this task. "
                        f"Tool groups for this task: {', '.join(tool_groups) if tool_groups else 'none'}"
                    )
                    tqdm.write(warning)
                    notes_messages.append(warning)
                    continue

                existing_entries = playbook.get_section_entries(section)
                try:
                    if op_type == "ADD":
                        content = operation.get("content", "")
                        if not content or not content.strip():
                            warning = (
                                f"[Warning] Skipping ADD operation for {entry_id} in section '{section}': "
                                f"content is missing or empty"
                            )
                            tqdm.write(warning)
                            notes_messages.append(warning)
                            continue
                        playbook.add_entry(section, content)
                        applied_any = True
                        applied_operations.append(
                            {
                                "type": "ADD",
                                "section": section,
                                "normalized_section": normalized_section,
                            }
                        )
                    elif op_type == "MODIFY":
                        operation_entry_id = operation.get("ID")
                        content = operation.get("content", "")
                        if not operation_entry_id:
                            warning = (
                                f"[Warning] Skipping MODIFY operation for {entry_id} in section '{section}': "
                                f"missing 'ID' field"
                            )
                            tqdm.write(warning)
                            notes_messages.append(warning)
                            continue
                        if not content or not content.strip():
                            warning = (
                                f"[Warning] Skipping MODIFY operation for {entry_id} in section '{section}': "
                                f"content is missing or empty"
                            )
                            tqdm.write(warning)
                            notes_messages.append(warning)
                            continue
                        if operation_entry_id in existing_entries:
                            playbook.modify_entry(section, operation_entry_id, content)
                            applied_operations.append(
                                {
                                    "type": "MODIFY",
                                    "section": section,
                                    "normalized_section": normalized_section,
                                }
                            )
                        else:
                            info_msg = (
                                f"[Info] MODIFY operation for {entry_id}: entry '{operation_entry_id}' not found in section '{section}'. "
                                f"Treating as ADD instead."
                            )
                            tqdm.write(info_msg)
                            notes_messages.append(info_msg)
                            playbook.add_entry(
                                section, content, entry_id=operation_entry_id
                            )
                            applied_operations.append(
                                {
                                    "type": "ADD",
                                    "section": section,
                                    "normalized_section": normalized_section,
                                }
                            )
                        applied_any = True
                    elif op_type == "REMOVE":
                        operation_entry_id = operation.get("ID")
                        if not operation_entry_id:
                            warning = (
                                f"[Warning] Skipping REMOVE operation for {entry_id} in section '{section}': "
                                f"missing 'ID' field"
                            )
                            tqdm.write(warning)
                            notes_messages.append(warning)
                            continue
                        if operation_entry_id in existing_entries:
                            playbook.remove_entry(section, operation_entry_id)
                            applied_any = True
                            applied_operations.append(
                                {
                                    "type": "REMOVE",
                                    "section": section,
                                    "normalized_section": normalized_section,
                                }
                            )
                        else:
                            warning = (
                                f"[Warning] Skipping REMOVE operation for {entry_id}: entry '{operation_entry_id}' "
                                f"not found in section '{section}'"
                            )
                            tqdm.write(warning)
                            notes_messages.append(warning)
                    else:
                        warning = (
                            f"[Warning] Skipping unknown operation type '{op_type}' for {entry_id} in section '{section}'"
                        )
                        tqdm.write(warning)
                        notes_messages.append(warning)
                except KeyError as exc:
                    warning = f"[Warning] Skipping invalid curator op for {entry_id}: {exc}"
                    tqdm.write(warning)
                    notes_messages.append(warning)
                if applied_any:
                    existing_entries = playbook.get_section_entries(section)

            outcome = "applied" if applied_any else "no_update"
            notes_messages.append(f"evaluation_status={evaluation.get('status')}")
            metrics_collector.record_sample(
                entry_id=entry_id,
                tool_groups=tool_groups,
                applied_operations=applied_operations,
                outcome=outcome,
                evaluation_passed=evaluation_valid,
                notes=" | ".join(notes_messages) if notes_messages else None,
            )
            if applied_any:
                playbook.save()
    finally:
        run_metadata["samples_recorded"] = len(metrics_collector.sample_records)
        shutil.rmtree(temp_result_root, ignore_errors=True)
        saved_paths = metrics_collector.save(metadata=run_metadata)
        if saved_paths.get("data_path"):
            tqdm.write(f"[Metrics] Training metrics saved to {saved_paths['data_path']}")
        if saved_paths.get("plot_path"):
            tqdm.write(f"[Metrics] Training plot saved to {saved_paths['plot_path']}")
        if is_oss_model:
            handler.shutdown_local_server()

