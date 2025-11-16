#!/usr/bin/env python3
"""
Calculate the start_offset for ACE training based on completed entries.

This script checks which training entries have already been processed
(by looking for completed prompt log files) and returns the offset
to resume training from where it left off.
"""

import random
from pathlib import Path

from bfcl_eval.ace.split_manager import (
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_SEED,
    TRAIN_PARTITION,
    ensure_split_exists,
    get_partition_ids,
)


def calculate_training_offset(
    prompt_log_dir: Path,
    split_path: Path = DEFAULT_SPLIT_PATH,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> int:
    """
    Calculate the number of training entries that have been completed.
    
    An entry is considered completed if it has all three required files:
    - generator_prompt.json
    - reflector_prompt.json
    - curator_prompt.json
    
    Returns the offset (number of completed entries) to use for --start-offset.
    """
    # Get training IDs in the same shuffled order as training would use
    ensure_split_exists(seed=split_seed, output_path=split_path, regenerate=False)
    train_ids = sorted(get_partition_ids(TRAIN_PARTITION, path=split_path))
    rng = random.Random(split_seed)
    rng.shuffle(train_ids)
    
    if not prompt_log_dir.exists():
        return 0
    
    training_dir = prompt_log_dir / "training"
    if not training_dir.exists():
        return 0
    
    # Check which entries have been completed
    completed_count = 0
    required_files = [
        "generator_prompt.json",
        "reflector_prompt.json",
        "curator_prompt.json",
    ]
    
    for entry_id in train_ids:
        entry_dir = training_dir / entry_id
        if not entry_dir.exists():
            break
        
        # Check if all required files exist
        if all((entry_dir / filename).exists() for filename in required_files):
            completed_count += 1
        else:
            # If any required file is missing, this entry is incomplete
            break
    
    return completed_count


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate start_offset for resuming ACE training"
    )
    parser.add_argument(
        "--prompt-log-dir",
        type=str,
        required=True,
        help="Path to the prompt log directory (should contain 'training' subdirectory)",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to the dataset split JSON (defaults to default split path)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed used for dataset split (must match training seed)",
    )
    
    args = parser.parse_args()
    
    prompt_log_dir = Path(args.prompt_log_dir)
    if not prompt_log_dir.is_absolute():
        from bfcl_eval.constants.eval_config import PROJECT_ROOT
        prompt_log_dir = PROJECT_ROOT / prompt_log_dir
    
    split_path = Path(args.split_path) if args.split_path else DEFAULT_SPLIT_PATH
    if not split_path.is_absolute():
        from bfcl_eval.constants.eval_config import PROJECT_ROOT
        split_path = PROJECT_ROOT / split_path
    
    offset = calculate_training_offset(
        prompt_log_dir=prompt_log_dir,
        split_path=split_path,
        split_seed=args.split_seed,
    )
    
    print(f"Completed entries: {offset}")
    print(f"Use --start-offset {offset} to resume training")


if __name__ == "__main__":
    main()

