#!/usr/bin/env bash
#SBATCH -A mt
#SBATCH -p mt
#SBATCH --job-name=train_v2g
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --output=job_%j_out.log
#SBATCH --error=job_%j_err.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/video_to_glosses"

# Replace these defaults with your local paths or export the env vars before submitting.
export CSLR_INPUT_DIR="${CSLR_INPUT_DIR:-/path/to/gloss_experiment/all_glosses}"
export CSLR_OUTPUT_DIR="${CSLR_OUTPUT_DIR:-$SCRIPT_DIR/output}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/path/to/venv/bin/activate}"

# Activate your environment
source "$VENV_ACTIVATE"

# Force unbuffered real-time tracking logs
python3 -u train.py