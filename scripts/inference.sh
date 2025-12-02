#!/bin/bash
#SBATCH -A bbjs-dtai-gh
#SBATCH -J bulk_inference
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 72
#SBATCH --mem=120G
#SBATCH -t 48:00:00
#SBATCH -o exp/inference_logs/%j.out
#SBATCH -e exp/inference_logs/%j.out

# run with
# sbatch -J inf scripts/inference.sh 
# [args for main.py]

# === Directory Setup ===
cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench
mkdir -p "exp/inference_logs"
# === Environment setup ===
source ~/.bashrc
conda deactivate
source setup_uv.sh .venv_dai
# for w2v2ph these must be pre-built
export PHONEMIZER_ESPEAK_LIBRARY="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/espeak-ng-data"
###########################

python src/main.py experiment=inference/buckeye_pr_powsm "$@"

# W2v2ph model options:
# experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft
# experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft
# experiment=inference/buckeye_pr_w2v2ph inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
# experiment=inference/buckeye_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k
# experiment=inference/buckeye_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k

# with masking
# data.mask_probability=0.x
# for mp in 0.0 0.2 0.4 0.6 0.8; do sbatch scripts/inference.sh data.mask_probability=$mp ; done

# on timit
# experiment=inference/timit_pr_powsm
# experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft
# experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft
# experiment=inference/timit_pr_w2v2ph inference.inference_runner.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
# experiment=inference/timit_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-500k
# experiment=inference/timit_pr_zipactc inference.inference_runner.hf_repo=anyspeech/zipa-large-crctc-ns-800k