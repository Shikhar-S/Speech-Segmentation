#!/bin/bash
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4
#SBATCH -J train
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o exp/slurm_logs/%x/%j.out
#SBATCH -e exp/slurm_logs/%x/%j.out

# run with
# sbatch -J train scripts/train.sh 
# [args for main.py]

# === Directory Setup ===
cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench
mkdir -p "exp/slurm_logs/${SLURM_JOB_NAME}"
# === Environment setup ===
source ~/.bashrc
conda deactivate
source setup_uv.sh .venv
# for w2v2ph these must be pre-built
export PHONEMIZER_ESPEAK_LIBRARY="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/espeak-ng-data"
###########################

python src/main.py experiment=probing/fa_buckeye_powsm "$@"