#!/bin/bash
# Train CTC head on TIMIT with WavLM-large backbone (fully fine-tuned),
# checkpoint-selected for RECOGNITION (monitor val_seg/per, mode=min).
#
# Submit:
#   sbatch scripts/run_wavlm_ctc_recognition_train.sh
#   sbatch scripts/run_wavlm_ctc_recognition_train.sh trainer.max_steps=30000   # override
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J wavlm-ctc-rec
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH -t 8:00:00
#SBATCH -o exp/slurm_logs/%j.out
#SBATCH -e exp/slurm_logs/%j.out


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

source scripts/env.sh

export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[setup] node=$(hostname) gpus=$(nvidia-smi -L | wc -l)"

srun python src/main.py \
  experiment=train/wavlm_ctc_recognition \
  "$@"
