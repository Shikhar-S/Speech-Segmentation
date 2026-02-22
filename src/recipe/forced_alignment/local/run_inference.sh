#!/bin/bash
#SBATCH -A bbjs-dtai-gh
#SBATCH -p ghx4
#SBATCH -J fa_inf
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 72
#SBATCH --mem=120G
#SBATCH -t 48:00:00
#SBATCH -o /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/fa_inf/%x_%j.out
#SBATCH -e /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/fa_inf/%x_%j.out

mkdir -p /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/fa_inf
# === Environment setup ===
source ~/.bashrc
conda deactivate
cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench
source setup_uv.sh .venv_dai

# run with
# sbatch /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/src/recipe/forced_alignment/local/run_inference.sh
# [args]

# for w2v2ph
export PHONEMIZER_ESPEAK_LIBRARY="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/espeak-ng-data"

# Default - powsm
python src/main.py experiment=inference/fa_powsm "$@" 

# For W2v2ph use:
# inference/fa_w2v2ph

# W2v2ph model options:
# data.model_tokenizer.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft inference.inference_runner.model.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft
# data.model_tokenizer.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns inference.inference_runner.model.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
# data.model_tokenizer.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft inference.inference_runner.model.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft
