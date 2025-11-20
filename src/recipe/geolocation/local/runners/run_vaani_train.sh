#!/bin/bash
#SBATCH -A bbjs-dtai-gh
#SBATCH -p ghx4
#SBATCH -J geovaani
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 72
#SBATCH --mem=120G
#SBATCH -t 48:00:00
#SBATCH -o /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/vaani_geolocation/%x_%j.out
#SBATCH -e /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/vaani_geolocation/%x_%j.out

mkdir -p /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/exp/vaani_geolocation
# === Environment setup ===
source ~/.bashrc
# conda activate powsmesp
conda deactivate
cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench
source setup_uv.sh .venv_dai

# run with
# sbatch /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/src/recipe/geolocation/local/runners/run_vaani_train.sh [args]

# for w2v2ph
export PHONEMIZER_ESPEAK_LIBRARY="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/src/.libs/libespeak-ng.so.1.1.51"
export ESPEAK_DATA_PATH="/work/nvme/bbjs/sbharadwaj/powsm/dai_dependencies/espeak-ng/espeak-ng-data"
# python src/main.py experiment=vaani_geolocation logger.wandb.tags=['140cluster','geolocation'] model.freeze_encoder=False "$@"

# freeze
python src/main.py experiment=vaani_geolocation logger.wandb.tags=['probe','140cluster','geolocation'] "$@"

# W2v2ph model options:
# +logger.wandb.name=F.lv-60 model=w2v2ph_geolocation model.model.hf_repo=facebook/wav2vec2-lv-60-espeak-cv-ft
# +logger.wandb.name=F.xlsr-53 model=w2v2ph_geolocation model.model.hf_repo=facebook/wav2vec2-xlsr-53-espeak-cv-ft
# +logger.wandb.name=F.ctaguchi model=w2v2ph_geolocation model.model.hf_repo=ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns

# for naive frontend model
# +logger.wandb.name=frontend_only model=naive_geolocation