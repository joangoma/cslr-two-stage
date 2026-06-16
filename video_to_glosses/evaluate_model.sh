#!/usr/bin/env bash
#SBATCH -A mt
#SBATCH -p mt
#SBATCH --job-name=eval_v2g
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB
#SBATCH --output=job_%j_out.log
#SBATCH --error=job_%j_err.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/video_to_glosses"

# Replace these defaults with your local paths or export the env vars before submitting.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/best_cslr_checkpoint.pt}"
DATA_FILE="${DATA_FILE:-/path/to/val.json}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/path/to/venv/bin/activate}"

# Activate your environment
source "$VENV_ACTIVATE"

# Force unbuffered real-time tracking logs
python3 -u evaluate.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_file "$DATA_FILE" \
    --save_predictions test_predictions.jsonl