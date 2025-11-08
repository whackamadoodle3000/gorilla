from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .constants import DEFAULT_PLAYBOOK_PATH, PLAYBOOK_PROMPT_HEADER
from .utils import camel_to_snake, list_tool_groups


@dataclass
class PlaybookOperationResult:
    section: str
    entry_id: str
    content: Optional[str] = None


class PlaybookManager:
    """
    Handles loading, updating, and rendering the ACE playbook.
    """

    def __init__(self, path: str | Path = DEFAULT_PLAYBOOK_PATH, reset: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, str]] = {}
        if reset and self.path.exists():
            self.path.unlink()
        if self.path.exists():
            self._load()
        else:
            self._initialize_empty()
            self.save()

    def _initialize_empty(self) -> None:
        self._data = {group: {} for group in list_tool_groups()}

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "playbook" in payload:
            sections = payload["playbook"]
        elif isinstance(payload, list):
            sections = payload
        else:
            raise ValueError(
                f"Unexpected playbook format in {self.path}. Expected a list of sections."
            )

        data: Dict[str, Dict[str, str]] = {}
        for section in sections:
            name = section.get("name") or section.get("Name")
            if not name:
                continue
            entries = section.get("entries", {})
            data[camel_to_snake(name)] = dict(entries)

        # Ensure all known tool groups are represented
        for group in list_tool_groups():
            data.setdefault(group, {})
        self._data = data

    def save(self) -> None:
        serializable = [
            {"Name": name, "entries": entries}
            for name, entries in sorted(self._data.items())
        ]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ---- CRUD helpers -----------------------------------------------------

    def add_entry(
        self, section: str, content: str, entry_id: str | None = None
    ) -> PlaybookOperationResult:
        normalized_section = camel_to_snake(section)
        entries = self._data.setdefault(normalized_section, {})
        if entry_id is None:
            entry_id = self._generate_entry_id(normalized_section)
        entries[entry_id] = content.strip()
        return PlaybookOperationResult(section=normalized_section, entry_id=entry_id, content=content.strip())

    def modify_entry(
        self, section: str, entry_id: str, content: str
    ) -> PlaybookOperationResult:
        normalized_section = camel_to_snake(section)
        entries = self._data.setdefault(normalized_section, {})
        if entry_id not in entries:
            raise KeyError(
                f"Cannot modify entry '{entry_id}' in section '{normalized_section}'; entry not found."
            )
        entries[entry_id] = content.strip()
        return PlaybookOperationResult(section=normalized_section, entry_id=entry_id, content=content.strip())

    def remove_entry(self, section: str, entry_id: str) -> PlaybookOperationResult:
        normalized_section = camel_to_snake(section)
        entries = self._data.setdefault(normalized_section, {})
        if entry_id not in entries:
            raise KeyError(
                f"Cannot remove entry '{entry_id}' in section '{normalized_section}'; entry not found."
            )
        removed = entries.pop(entry_id)
        return PlaybookOperationResult(section=normalized_section, entry_id=entry_id, content=removed)

    def get_section_entries(self, section: str) -> Dict[str, str]:
        return dict(self._data.get(camel_to_snake(section), {}))

    def to_prompt_string(self) -> str:
        if all(len(entries) == 0 for entries in self._data.values()):
            return f"{PLAYBOOK_PROMPT_HEADER}\nNo insights recorded yet."

        body_lines: List[str] = []
        for section, entries in sorted(self._data.items()):
            if not entries:
                continue
            body_lines.append(f"{section}:")
            for entry_id, content in sorted(entries.items()):
                body_lines.append(f"- {entry_id}: {content}")
            body_lines.append("")  # blank line between sections
        json_view = json.dumps(
            [
                {"Name": section, "entries": entries}
                for section, entries in sorted(self._data.items())
            ],
            ensure_ascii=False,
            indent=2,
        )
        joined = "\n".join(line for line in body_lines if line is not None)
        return f"{PLAYBOOK_PROMPT_HEADER}\n\n{joined.strip()}\n\nRaw JSON:\n{json_view}"

    def _generate_entry_id(self, section: str) -> str:
        entries = self._data.setdefault(section, {})
        existing_numbers = []
        for entry_id in entries.keys():
            if entry_id.startswith("delta"):
                suffix = entry_id[5:]
                if suffix.isdigit():
                    existing_numbers.append(int(suffix))
        next_number = max(existing_numbers, default=0) + 1
        return f"delta{next_number}"

    def sections(self) -> Iterable[str]:
        return self._data.keys()

    def data(self) -> Dict[str, Dict[str, str]]:
        return self._data

