#!/bin/bash
#SBATCH -A bbjs-dtai-gh
#SBATCH -p ghx4
#SBATCH -J train
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 72
#SBATCH --mem=120G
#SBATCH -t 48:00:00
#SBATCH -o exp/inf_%x/%j.out
#SBATCH -e exp/inf_%x/%j.out

# run with
# sbatch -J inf scripts/inference.sh 
# [args for main.py]

# === Directory Setup ===
cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench
mkdir -p "exp/inf_${SLURM_JOB_NAME}"
# === Environment setup ===
source ~/.bashrc
conda deactivate
source setup_uv.sh .venv_dai
# for w2v2ph these must be pre-built
export PHONEMIZER_ESPEAK_LIBRARY="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/espeak-ng-data"
###########################

python src/main.py experiment=inference/pr_powsm "$@"

# W2v2ph model options:
# experiment=inference/pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft
# experiment=inference/pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft
# experiment=inference/pr_w2v2ph inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns