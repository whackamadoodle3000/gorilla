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

