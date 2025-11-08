from __future__ import annotations

from pathlib import Path

from bfcl_eval.constants.eval_config import PROJECT_ROOT

ACE_DATA_DIR = PROJECT_ROOT / "bfcl_eval" / "data" / "ace"
ACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PLAYBOOK_PATH = ACE_DATA_DIR / "playbook.json"
DEFAULT_SPLIT_PATH = ACE_DATA_DIR / "dataset_split.json"

DEFAULT_SPLIT_SEED = 2025
DEFAULT_TRAIN_RATIO = 0.8

PLAYBOOK_PROMPT_HEADER = (
    "ACE Playbook Insights\n"
    "Use these distilled reminders to plan tool usage. Keep tool calls faithful to the "
    "functions provided in the BFCL benchmark."
)

ACE_PLAYBOOK_SYSTEM_MARKER = "[ACE_PLAYBOOK]"

