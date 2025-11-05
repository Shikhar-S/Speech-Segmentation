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
conda activate powsmesp
cd /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench

# run with
# sbatch /work/nvme/bbjs/sbharadwaj/powsm/PhoneBench/src/recipe/geolocation/local/runners/run_vaani_train.sh [args]
python src/main.py experiment=vaani_geolocation "$@"