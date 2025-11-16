"""Utility to clean BFCL result JSONL files by removing duplicates and failed generations.

This script emulates the manual fix we applied for DeepSeek results after a
multithreaded run produced duplicate entries and 404 responses. It walks every
`*.json` file beneath the provided result root, discards entries whose `result`
field contains the 404 error sentinel, and keeps only the first instance of each
`id`. For multi-turn categories it additionally checks that the decoded result
has the same number of turns as the ground-truth trace—mismatched entries are
skipped so evaluation does not fail.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

from bfcl_eval.constants.category_mapping import MULTI_TURN_CATEGORY
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry


def _category_from_filename(path: Path) -> str:
    """Extract the bare category name from a result file path."""
    stem = path.stem
    # Files follow the pattern BFCL_v4_<category>_result.json
    return stem.replace("BFCL_v4_", "").replace("_result", "")


def _iter_clean_entries(
    path: Path,
    category: str,
    prompts_by_id: Dict[str, dict],
    ground_truth_by_id: Dict[str, list] | None,
) -> Iterable[Tuple[str, dict]]:
    """Yield cleaned (id, entry) pairs from a JSONL result file."""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entry = json.loads(line)
        entry_id = entry["id"]

        result_payload = entry.get("result")
        if isinstance(result_payload, str) and result_payload.startswith(
            "Error during inference: Error code: 404"
        ):
            continue

        # Skip entries that no longer line up with the official prompts list.
        if entry_id not in prompts_by_id:
            continue

        if ground_truth_by_id is not None:
            ground_truth_turns = ground_truth_by_id.get(entry_id)
            if ground_truth_turns is None:
                continue
            if not isinstance(result_payload, list) or len(result_payload) != len(
                ground_truth_turns
            ):
                continue

        yield entry_id, entry


def clean_file(path: Path) -> Tuple[int, int, int]:
    """Deduplicate + filter a single JSONL file. Returns stats tuple."""
    category = _category_from_filename(path)
    prompt_entries = load_dataset_entry(category)
    prompts_by_id = {entry["id"]: entry for entry in prompt_entries}

    ground_truth_by_id = None
    if category in MULTI_TURN_CATEGORY:
        gt_entries = load_ground_truth_entry(category)
        ground_truth_by_id = {
            prompt_entries[idx]["id"]: gt_entries[idx]["ground_truth"]
            for idx in range(len(prompt_entries))
        }

    seen: Dict[str, dict] = {}
    for entry_id, entry in _iter_clean_entries(path, category, prompts_by_id, ground_truth_by_id):
        # Only keep the first occurrence of each id.
        if entry_id not in seen:
            seen[entry_id] = entry

    cleaned_lines = [json.dumps(seen[key], ensure_ascii=False) for key in sorted(seen)]
    # Ensure trailing newline for consistent appends.
    path.write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""))

    return len(seen), len(prompt_entries), len(cleaned_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate/error entries from BFCL result JSONL files."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("result"),
        help="Directory containing per-model result folders (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional subdirectory under results-root to clean "
            "(e.g. 'DeepSeek-V3.2-Exp-FC'). If omitted, every JSON file under the root is processed."
        ),
    )

    args = parser.parse_args()
    root = args.results_root
    if args.model:
        root = root / args.model

    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"No JSON result files found under {root}")
        return

    summary: Dict[str, Counter] = {}
    for file_path in files:
        kept, prompt_count, _ = clean_file(file_path)
        summary[str(file_path)] = Counter(kept=kept, expected=prompt_count)
        print(f"{file_path}: kept {kept} entries (expected {prompt_count})")

    print("\nSummary:")
    for file_path, counter in summary.items():
        print(f"- {file_path}: {counter['kept']} / {counter['expected']}")


if __name__ == "__main__":
    main()

