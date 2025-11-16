from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

from bfcl_eval.constants.category_mapping import ALL_CATEGORIES
from bfcl_eval.utils import load_dataset_entry

from .constants import (
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
)
from .utils import determine_tool_groups, list_tool_groups


TRAIN_PARTITION = "train"
TEST_PARTITION = "test"


@dataclass
class DatasetEntry:
    id: str
    category: str
    groups: List[str]


def _load_all_entries(
    include_prereq: bool = False,
    exclude_categories: Sequence[str] | None = None,
) -> List[DatasetEntry]:
    entries: List[DatasetEntry] = []
    excluded = set(exclude_categories or [])
    for category in ALL_CATEGORIES:
        if category in excluded:
            continue
        prompt_entries = load_dataset_entry(
            category,
            include_prereq=include_prereq,
            include_language_specific_hint=False,
        )
        for entry in prompt_entries:
            entry_id = entry["id"]
            # Skip prerequisite entries – they mirror write-only steps
            if entry_id.endswith("_prereq"):
                continue
            groups = determine_tool_groups(entry)
            entries.append(DatasetEntry(id=entry_id, category=category, groups=groups))
    return entries


def _ensure_group_coverage(
    entries: Sequence[DatasetEntry],
    rng: random.Random,
) -> Dict[str, Set[str]]:
    group_to_entries: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        for group in entry.groups:
            group_to_entries[group].append(entry.id)

    for group in list_tool_groups():
        group_to_entries.setdefault(group, [])

    coverage = {
        TRAIN_PARTITION: set(),
        TEST_PARTITION: set(),
    }

    for group, ids in group_to_entries.items():
        if not ids:
            continue
        ids = list(dict.fromkeys(ids))
        rng.shuffle(ids)

        available_for_train = [
            candidate
            for candidate in ids
            if candidate not in coverage[TRAIN_PARTITION] and candidate not in coverage[TEST_PARTITION]
        ]
        if available_for_train:
            coverage[TRAIN_PARTITION].add(available_for_train[0])

        available_for_test = [
            candidate
            for candidate in ids
            if candidate not in coverage[TRAIN_PARTITION]
            and candidate not in coverage[TEST_PARTITION]
        ]
        if available_for_test:
            coverage[TEST_PARTITION].add(available_for_test[0])
    return coverage


def _assign_remaining(
    entries: Sequence[DatasetEntry],
    coverage: Dict[str, Set[str]],
    rng: random.Random,
    train_ratio: float,
) -> None:
    # Determine desired train size (rounded)
    unique_ids = list({entry.id for entry in entries})
    rng.shuffle(unique_ids)
    target_train_size = max(1, int(len(unique_ids) * train_ratio))

    for entry_id in unique_ids:
        in_train = entry_id in coverage[TRAIN_PARTITION]
        in_test = entry_id in coverage[TEST_PARTITION]
        if in_train or in_test:
            continue
        if len(coverage[TRAIN_PARTITION]) < target_train_size:
            coverage[TRAIN_PARTITION].add(entry_id)
        else:
            coverage[TEST_PARTITION].add(entry_id)

    # Guard against empty test partition due to rounding
    if not coverage[TEST_PARTITION]:
        last_id = unique_ids[-1]
        if last_id in coverage[TRAIN_PARTITION]:
            coverage[TRAIN_PARTITION].remove(last_id)
        coverage[TEST_PARTITION].add(last_id)


def _build_split_structure(
    entries: Sequence[DatasetEntry],
    coverage: Mapping[str, Set[str]],
    seed: int,
    train_ratio: float,
) -> Dict:
    per_category = defaultdict(lambda: {TRAIN_PARTITION: [], TEST_PARTITION: []})
    for entry in entries:
        if entry.id in coverage[TRAIN_PARTITION]:
            per_category[entry.category][TRAIN_PARTITION].append(entry.id)
        if entry.id in coverage[TEST_PARTITION]:
            per_category[entry.category][TEST_PARTITION].append(entry.id)

    # Sort identifiers for determinism
    for category_data in per_category.values():
        category_data[TRAIN_PARTITION].sort()
        category_data[TEST_PARTITION].sort()

    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "train_ratio": train_ratio,
            "total_entries": len(entries),
        },
        "partitions": {
            TRAIN_PARTITION: sorted(coverage[TRAIN_PARTITION]),
            TEST_PARTITION: sorted(coverage[TEST_PARTITION]),
        },
        "per_category": dict(per_category),
    }


def ensure_split_exists(
    seed: int = DEFAULT_SPLIT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    output_path: Path = DEFAULT_SPLIT_PATH,
    regenerate: bool = False,
    exclude_categories: Sequence[str] | None = None,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not regenerate:
        return output_path

    rng = random.Random(seed)
    entries = _load_all_entries(exclude_categories=exclude_categories)

    coverage = _ensure_group_coverage(entries, rng)
    _assign_remaining(entries, coverage, rng, train_ratio)
    structure = _build_split_structure(entries, coverage, seed, train_ratio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)
    return output_path


def _load_split(path: Path = DEFAULT_SPLIT_PATH) -> Dict:
    ensure_split_exists(output_path=path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_partition_ids(
    partition: str,
    path: Path = DEFAULT_SPLIT_PATH,
) -> Set[str]:
    split_data = _load_split(path)
    try:
        ids = split_data["partitions"][partition]
    except KeyError as err:
        raise KeyError(f"Unknown dataset partition '{partition}'") from err
    return set(ids)


def summarize_split(path: Path = DEFAULT_SPLIT_PATH) -> Dict[str, Dict[str, int]]:
    split_data = _load_split(path)
    per_category = split_data.get("per_category", {})
    summary = {}
    for category, partitions in per_category.items():
        summary[category] = {
            partition: len(ids) for partition, ids in partitions.items()
        }
    return summary

