#!/usr/bin/env bash
#SBATCH -A mt
#SBATCH -p mt
#SBATCH --job-name=train_gloss_slt
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --output=job_%j_out.log
#SBATCH --error=job_%j_err.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Replace these defaults with your local paths or export the env vars before submitting.
export G2T_DATA_DIR="${G2T_DATA_DIR:-/path/to/ca_glosses}"
export G2T_OUTPUT_DIR="${G2T_OUTPUT_DIR:-$SCRIPT_DIR/output}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/path/to/venv/bin/activate}"

# Activate your environment
source "$VENV_ACTIVATE"

# Force unbuffered real-time tracking logs
python3 -u train_gloss_to_text_baseline.py