#!/bin/bash
# Submit BCE boundary inference for both models on all 3 datasets.
# Usage: bash scripts/bce_loss/run_bce_inference.sh

set -euo pipefail

XEUS_CKPT="exp/runs/seg_bce_xeus/20260323_194645/checkpoints/step_000562.ckpt"
PXEUS_CKPT="exp/runs/seg_bce_pxeus/20260323_194652/checkpoints/step_000562.ckpt"

declare -A DATASETS
DATASETS[timit]=changelinglab/timit-segment
DATASETS[buckeye]=changelinglab/buckeye-segment
DATASETS[voxangeles]=changelinglab/voxangeles-segment

for dataset in timit buckeye voxangeles; do
    hf_repo="${DATASETS[$dataset]}"

    # XEUS
    sbatch scripts/bce_loss/inference_xeus.batch \
        inference.inference_runner.ckpt_path="${XEUS_CKPT}" \
        data.hf_repo="${hf_repo}" \
        task_name="inf_bce_xeus_${dataset}" \
        run_folder="inf_bce_xeus_${dataset}"

    # pXEUS
    sbatch scripts/bce_loss/inference_pxeus.batch \
        inference.inference_runner.ckpt_path="${PXEUS_CKPT}" \
        data.hf_repo="${hf_repo}" \
        task_name="inf_bce_pxeus_${dataset}" \
        run_folder="inf_bce_pxeus_${dataset}"
done
