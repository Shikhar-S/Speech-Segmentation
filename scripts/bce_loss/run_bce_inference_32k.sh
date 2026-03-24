#!/bin/bash
# Submit BCE boundary inference (32kHz) for both models on all 3 datasets.
# Usage: bash scripts/bce_loss/run_bce_inference_32k.sh

set -euo pipefail

XEUS_CKPT="exp/runs/seg_bce_xeus_32k/FILL_DATE/checkpoints/last.ckpt"
PXEUS_CKPT="exp/runs/seg_bce_pxeus_32k/FILL_DATE/checkpoints/last.ckpt"

declare -A DATASETS
DATASETS[timit]=changelinglab/timit-segment
DATASETS[buckeye]=changelinglab/buckeye-segment
DATASETS[voxangeles]=changelinglab/voxangeles-segment

for dataset in timit buckeye voxangeles; do
    hf_repo="${DATASETS[$dataset]}"

    # XEUS
    sbatch scripts/bce_loss/inference_xeus_32k.batch \
        inference.inference_runner.ckpt_path="${XEUS_CKPT}" \
        data.hf_repo="${hf_repo}" \
        task_name="inf_bce_xeus_32k_${dataset}" \
        run_folder="inf_bce_xeus_32k_${dataset}"

    # pXEUS
    sbatch scripts/bce_loss/inference_pxeus_32k.batch \
        inference.inference_runner.ckpt_path="${PXEUS_CKPT}" \
        data.hf_repo="${hf_repo}" \
        task_name="inf_bce_pxeus_32k_${dataset}" \
        run_folder="inf_bce_pxeus_32k_${dataset}"
done
