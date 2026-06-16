#!/bin/bash
#SBATCH -A mt
#SBATCH -p mt
#SBATCH --job-name=metrics-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4      
#SBATCH --mem=40G
#SBATCH --output=cascade_eval_%j.out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_ACTIVATE="${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
V2G_CHECKPOINT="${V2G_CHECKPOINT:-/path/to/best_model.pt}"
G2T_MODEL_DIR="${G2T_MODEL_DIR:-/path/to/gloss_to_text/final_model}"
DATA_FILE="${DATA_FILE:-/path/to/test.json}"
VOCAB_PATH="${VOCAB_PATH:-/path/to/vocab.json}"

source "$VENV_ACTIVATE"

python cascade_evaluate.py \
    --v2g_checkpoint "$V2G_CHECKPOINT" \
    --g2t_model_dir "$G2T_MODEL_DIR" \
    --data_file "$DATA_FILE" \
    --vocab_path "$VOCAB_PATH" \
    --translation_field translation \
    --batch_size_g2t 16 \
    --num_beams 10 \
    --save_predictions predictions.json