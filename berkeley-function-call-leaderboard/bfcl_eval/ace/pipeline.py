from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from bfcl_eval._llm_response_generation import (
    build_handler,
    multi_threaded_inference,
)
from bfcl_eval.ace.playbook import PlaybookManager
from bfcl_eval.ace.split_manager import (
    TRAIN_PARTITION,
    ensure_split_exists,
    get_partition_ids,
)
from bfcl_eval.constants.category_mapping import ALL_CATEGORIES
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.utils import (
    extract_test_category_from_id,
    load_dataset_entry,
    load_ground_truth_entry,
)

from .constants import (
    DEFAULT_PLAYBOOK_PATH,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
)
from .llm import ChatCompletionClient
from .utils import determine_tool_groups, format_conversation, serialize_json


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

    @classmethod
    def build(cls) -> "DatasetCache":
        prompt: Dict[str, dict] = {}
        ground_truth: Dict[str, dict] = {}

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
        return cls(prompt=prompt, ground_truth=ground_truth)

    def get_prompt(self, entry_id: str) -> Optional[dict]:
        return self.prompt.get(entry_id)

    def get_ground_truth(self, entry_id: str) -> Optional[dict]:
        return self.ground_truth.get(entry_id)


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


def _build_reflector_messages(
    playbook_text: str,
    entry: dict,
    generator_result: dict,
    ground_truth: dict,
    tool_groups: Sequence[str],
) -> List[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "You are the ACE Reflector. Identify precise, short insights about why a "
            "model's tool usage differed from the BFCL ground truth. Respond in compact JSON."
        ),
    }
    user_content = f"""
Sample ID: {entry['id']}
Tool groups: {', '.join(tool_groups)}

Current playbook state:
{playbook_text}

Conversation:
{format_conversation(entry['question'])}

Model output:
{_summarize_generator_output(generator_result)}

Ground truth tool calls:
{serialize_json(ground_truth.get('ground_truth', {}))}

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
) -> List[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "You are the ACE Curator. Convert reflections into crisp playbook operations "
            "per tool group. Prefer modifying existing bullets when possible. Output JSON."
        ),
    }
    user_content = f"""
Sample ID: {entry['id']}
Tool groups: {', '.join(tool_groups)}

Existing playbook:
{playbook_text}

Model output:
{_summarize_generator_output(generator_result)}

Ground truth tool calls:
{serialize_json(ground_truth.get('ground_truth', {}))}

Reflection:
{serialize_json(reflection)}

Return JSON with keys "reasoning" (<=2 sentences) and "operations".
Each operation must be one of:
  {{"type": "ADD", "section": "<tool_group>", "content": "<short bullet>"}}
  {{"type": "MODIFY", "section": "<tool_group>", "ID": "<existing id>", "content": "<short bullet>"}}
  {{"type": "REMOVE", "section": "<tool_group>", "ID": "<existing id>"}}
Keep content under 160 characters. If no update is needed, return an empty list for operations.
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
    regenerate_split: bool = False,
    num_gpus: int = 1,
    backend: str = "sglang",
    gpu_memory_utilization: float = 0.9,
    skip_server_setup: bool = False,
    local_model_path: Optional[str] = None,
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
    if limit:
        train_ids = train_ids[:limit]

    playbook = PlaybookManager(playbook_path, reset=False)
    dataset_cache = DatasetCache.build()

    handler = build_handler(generator_model, generator_temperature)

    reflector_client = ChatCompletionClient(model=reflector_model, temperature=completion_temperature)
    curator_client = ChatCompletionClient(model=curator_model, temperature=completion_temperature)

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
                continue
            ground_truth = dataset_cache.get_ground_truth(entry_id)
            if not ground_truth:
                continue

            tool_groups = determine_tool_groups(entry)

            generator_result = multi_threaded_inference(
                handler=handler,
                test_case=deepcopy(entry),
                include_input_log=False,
                exclude_state_log=False,
                playbook_text=None,
            )

            playbook_text = playbook.to_prompt_string()
            reflection_messages = _build_reflector_messages(
                playbook_text,
                entry,
                generator_result,
                ground_truth,
                tool_groups,
            )
            reflection_raw = reflector_client.complete(reflection_messages, temperature=completion_temperature)
            reflection_raw_clean = _extract_json_block(reflection_raw)
            try:
                reflection_json = json.loads(reflection_raw_clean)
            except json.JSONDecodeError as exc:
                tqdm.write(f"[Warning] Failed to parse reflector output for {entry_id}: {exc}")
                continue

            curator_messages = _build_curator_messages(
                playbook_text,
                entry,
                generator_result,
                reflection_json,
                ground_truth,
                tool_groups,
            )
            curator_raw = curator_client.complete(curator_messages, temperature=completion_temperature)
            curator_raw_clean = _extract_json_block(curator_raw)
            try:
                curator_json = json.loads(curator_raw_clean)
            except json.JSONDecodeError as exc:
                tqdm.write(f"[Warning] Failed to parse curator output for {entry_id}: {exc}")
                continue

            operations = curator_json.get("operations", [])
            applied_any = False
            for operation in operations:
                op_type = operation.get("type", "").upper()
                section = operation.get("section") or (tool_groups[0] if tool_groups else None)
                if not section:
                    continue
                existing_entries = playbook.get_section_entries(section)
                try:
                    if op_type == "ADD":
                        content = operation.get("content", "")
                        playbook.add_entry(section, content)
                        applied_any = True
                    elif op_type == "MODIFY":
                        entry_id = operation.get("ID")
                        content = operation.get("content", "")
                        if entry_id:
                            if entry_id in existing_entries:
                                playbook.modify_entry(section, entry_id, content)
                            else:
                                # Fallback: treat as add when entry does not yet exist
                                playbook.add_entry(section, content, entry_id=entry_id)
                            applied_any = True
                    elif op_type == "REMOVE":
                        entry_id = operation.get("ID")
                        if entry_id:
                            if entry_id in existing_entries:
                                playbook.remove_entry(section, entry_id)
                                applied_any = True
                except KeyError as exc:
                    tqdm.write(f"[Warning] Skipping invalid curator op for {entry_id}: {exc}")
                # Refresh cached view after each operation to stay in sync
                if applied_any:
                    existing_entries = playbook.get_section_entries(section)
            if applied_any:
                playbook.save()
    finally:
        if is_oss_model:
            handler.shutdown_local_server()

