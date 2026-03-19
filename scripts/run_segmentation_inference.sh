#!/bin/bash

BASE=exp/runs/speech_segmentation

# Format: "step_filename|experiment_config"
declare -A RUNS

RUNS[seg_pxeus_frac0_053]="step_000875.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_106]="step_001500.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_2]="step_002250.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_4]="step_000500.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_6]="step_000500.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_8]="step_000625.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac1_0]="step_000625.ckpt|inference/segmentation_pxeus"

RUNS[seg_xeuspr_frac0_053]="step_000875.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_106]="step_000875.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_2]="step_000750.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_4]="step_000750.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_6]="step_000750.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_8]="step_001000.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac1_0]="step_001375.ckpt|inference/segmentation_xeuspr"

GREEDY=false

for run_folder in "${!RUNS[@]}"; do
    IFS='|' read -r ckpt_file exp <<< "${RUNS[$run_folder]}"
    ckpt=${BASE}/${run_folder}/checkpoints/${ckpt_file}
    
    sbatch scripts/daixpr_inference.batch \
        experiment=${exp} \
        run_folder=${run_folder} \
        inference.inference_runner.ckpt_path="${ckpt}" \
        inference.inference_runner.greedy=${GREEDY}
done
