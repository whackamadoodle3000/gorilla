from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH


GENERIC_GROUP_NAME = "generic"


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").lower()


def list_tool_groups() -> list[str]:
    groups = [
        path.stem
        for path in MULTI_TURN_FUNC_DOC_PATH.glob("*.json")
        if path.is_file()
    ]
    groups.sort()
    # generic bucket for categories that do not map cleanly
    if GENERIC_GROUP_NAME not in groups:
        groups.append(GENERIC_GROUP_NAME)
    return groups


def determine_tool_groups(entry: Mapping) -> list[str]:
    """
    Determine the tool groups associated with a dataset entry.
    """
    groups: set[str] = set()
    if "involved_classes" in entry:
        for class_name in entry["involved_classes"]:
            groups.add(camel_to_snake(class_name))
    elif "function" in entry and entry["function"]:
        for function_doc in entry["function"]:
            func_name = function_doc.get("name", "")
            prefix = func_name.split(".", 1)[0]
            if prefix:
                groups.add(camel_to_snake(prefix))
    if not groups:
        groups.add(GENERIC_GROUP_NAME)
    return sorted(groups)


def count_available_tools(entry: Mapping) -> int:
    """
    Estimate the number of tools available for a given dataset entry.
    """
    if not entry:
        return 0

    if "function" in entry and isinstance(entry["function"], list):
        return len(entry["function"])

    if "tools" in entry:
        tools = entry["tools"]
        if isinstance(tools, dict):
            return len(tools)
        if isinstance(tools, list):
            return len(tools)

    groups = determine_tool_groups(entry)
    return len(groups)


def extract_base_prompt_id(entry_id: str) -> str:
    """
    Normalize dataset entry IDs to a shared base form so we can align prompts with
    their published BFCL ground truths.

    Examples
    --------
    - format_sensitivity_0:...:simple_python_19 -> simple_python_19
    - memory_vector_131-notetaker-1 -> memory_vector_131-notetaker-1
      (no change because the evaluator itself aligns by index; we surface the raw ID for visibility)
    - web_search_base_42 -> web_search_42
    - web_search_no_snippet_5 -> web_search_5
    - live_relevance_* -> live_relevance_* (still needs checker logic)
    """
    if entry_id.startswith("format_sensitivity_"):
        # Format sensitivity IDs look like:
        # format_sensitivity_<seed>:...:<base_id>
        if ":" in entry_id:
            return entry_id.split(":")[-1]
    if entry_id.startswith("web_search_base_"):
        suffix = entry_id[len("web_search_base_") :]
        return f"web_search_{suffix}"
    if entry_id.startswith("web_search_no_snippet_"):
        suffix = entry_id[len("web_search_no_snippet_") :]
        return f"web_search_{suffix}"
    return entry_id


def extract_memory_ground_truth_id(entry_id: str) -> str | None:
    """
    BFCL's published ground-truth files for memory categories use the prefix 'memory_'.
    """
    if entry_id.startswith("memory_kv_"):
        return entry_id.replace("memory_kv_", "memory_", 1)
    if entry_id.startswith("memory_vector_"):
        return entry_id.replace("memory_vector_", "memory_", 1)
    if entry_id.startswith("memory_rec_sum_"):
        return entry_id.replace("memory_rec_sum_", "memory_", 1)
    return None


def serialize_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def format_conversation(conversation: Sequence[Sequence[Mapping]]) -> str:
    """
    Convert BFCL conversation turns into a readable JSON string.
    """
    return serialize_json(conversation)


def flatten_unique(items: Iterable[Iterable[str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in items:
        for value in group:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
    return ordered

