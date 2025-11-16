#!/bin/bash
# Run ACE training and testing workflow:
# 1. Extract 100 test IDs from multi-turn categories
# 2. Run ACE training on 100 samples
# 3. Run ACE testing on 100 IDs with custom output directory

set -e  # Exit on error

# Get the script directory (assumes script is run from berkeley-function-call-leaderboard)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Step 1: Extracting 100 test IDs"
echo "=========================================="
python -m bfcl_eval.scripts.extract_100_test_ids

echo ""
echo "=========================================="
echo "Step 2: Running ACE training on 100 samples"
echo "=========================================="

# Calculate how many entries have already been completed
START_OFFSET=$(python -m bfcl_eval.scripts.calculate_training_offset --prompt-log-dir "ace_prompts" 2>/dev/null | grep "Completed entries:" | awk '{print $3}' || echo "0")

if [ "$START_OFFSET" -gt "0" ]; then
  echo "[Info] Found $START_OFFSET completed entries. Resuming from offset $START_OFFSET..."
else
  echo "[Info] Starting fresh training..."
fi

python -m bfcl_eval ace-train-playbook \
  --generator-model "DeepSeek-V3.2-Exp-FC" \
  --reflector-model "deepseek-chat" \
  --curator-model "deepseek-chat" \
  --limit 100 \
  --start-offset "$START_OFFSET" \
  --generator-temperature 0.001 \
  --completion-temperature 0.0 \
  --prompt-log-dir "ace_prompts"

echo ""
echo "=========================================="
echo "Step 3: Running ACE testing on 100 test IDs"
echo "=========================================="
python -m bfcl_eval generate \
  --model "DeepSeek-V3.2-Exp-FC" \
  --test-category "all" \
  --ace \
  --run-ids \
  --ace-prompt-log-dir "ace_prompts" \
  --result-dir "result_ace_100_test" \
  --allow-overwrite \
  --temperature 0.001

echo ""
echo "=========================================="
echo "✅ Experiment complete!"
echo "=========================================="
echo "Results saved to: result_ace_100_test/"
echo "Prompts saved to: ace_prompts/"
echo "  - Training prompts: ace_prompts/training/"
echo "  - Testing prompts: ace_prompts/testing/"

