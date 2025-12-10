#!/bin/bash
# SLURM batch script for running wav2vec2-phoneme inference on CMU L2Arctic (L1 classification)
#
# Usage:
#   sbatch src/recipe/l1_classification/local/run_cmu_l2arctic_w2v2ph.sbatch \
#       [additional hydra overrides...]
#
# Customize SBATCH directives below as needed for the Babel partition/account you use.
# Defaults target a 1-GPU job in the "general" partition.

#SBATCH --job-name=cmu_l1_w2v2ph
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/slurm_logs/cmu_l2arctic_w2v2ph_%j.out
#SBATCH --error=/home/%u/slurm_logs/cmu_l2arctic_w2v2ph_%j.err

set -euo pipefail

LOG_DIR="/home/$USER/slurm_logs"
mkdir -p "$LOG_DIR"

# -----------------------------
# Hugging Face cache locations
# -----------------------------
export HF_HOME=${HF_HOME:-/data/user_data/$USER/.hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/data/hf_cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/hf_cache/datasets}
# Leave HF_HUB_OFFLINE unset (or 0) if you need to download artifacts
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# -----------------------------
# Project & Python environment
# -----------------------------
PROJECT_ROOT=${PROJECT_ROOT:-/home/$USER/PhoneBench}
VENV_DIR=${VENV_DIR:-.venv}

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "Project root not found: $PROJECT_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual env '$VENV_DIR' not found. Bootstrapping with setup_uv.sh..."
  bash setup_uv.sh "$VENV_DIR"
else
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT"

# -----------------------------
# Run inference
# -----------------------------
# Override e.g. inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft if needed.
python src/main.py experiment=inference/cmu_l2arctic_w2v2ph "$@"


