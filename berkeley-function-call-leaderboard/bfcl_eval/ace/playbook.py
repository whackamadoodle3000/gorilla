from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .constants import DEFAULT_PLAYBOOK_PATH, PLAYBOOK_PROMPT_HEADER
from .utils import camel_to_snake


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
        self._data = {}

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

    def to_prompt_string(
        self,
        focus_sections: Optional[Iterable[str]] = None,
        max_sections: Optional[int] = None,
        max_entries_per_section: Optional[int] = None,
        include_raw_json: bool = False,
    ) -> str:
        """
        Render the playbook for prompting. Focus on the provided sections (if any)
        and optionally truncate sections or entries to keep the output compact.
        """

        def _normalize(section_name: str) -> str:
            return camel_to_snake(section_name)

        focus_order: Optional[List[str]] = None
        if focus_sections:
            # Preserve caller order while normalizing and deduplicating
            seen: set[str] = set()
            focus_order = []
            for section in focus_sections:
                normalized = _normalize(section)
                if normalized not in seen:
                    focus_order.append(normalized)
                    seen.add(normalized)

        # Build ordered list of sections to render
        ordered_sections: List[str] = []
        if focus_order:
            ordered_sections.extend(focus_order)

        ordered_sections.extend(
            section
            for section in sorted(self._data.keys())
            if section not in ordered_sections
        )

        rendered_sections: List[tuple[str, List[tuple[str, str]]]] = []
        for section in ordered_sections:
            entries = self._data.get(section, {})
            if not entries:
                continue
            section_entries = sorted(entries.items())
            if max_entries_per_section is not None:
                section_entries = section_entries[:max_entries_per_section]
            rendered_sections.append((section, section_entries))
            if max_sections is not None and len(rendered_sections) >= max_sections:
                break

        header = PLAYBOOK_PROMPT_HEADER.rstrip()
        if not rendered_sections:
            return f"{header}\nNo insights recorded yet."

        body_lines: List[str] = [header, ""]
        for section, entries in rendered_sections:
            body_lines.append(section)
            for entry_id, content in entries:
                content_compact = " ".join(content.strip().split())
                body_lines.append(f"- {entry_id}: {content_compact}")
            body_lines.append("")

        output = "\n".join(line for line in body_lines if line is not None).strip()

        if include_raw_json:
            subset = [
                {"Name": section, "entries": dict(entries)}
                for section, entries in rendered_sections
            ]
            json_view = json.dumps(subset, ensure_ascii=False, separators=(",", ":"))
            output = f"{output}\n\nRaw JSON:{json_view}"

        return output

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

