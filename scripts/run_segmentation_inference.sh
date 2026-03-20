#!/bin/bash

GREEDY=true
DATASET=buckeye

BASE=exp/runs/speech_segmentation
INF_BASE=exp/runs/inf_segmentation

declare -A EXPECTED_SAMPLES
EXPECTED_SAMPLES[timit]=6300
EXPECTED_SAMPLES[buckeye]=10477
EXPECTED_SAMPLES[voxangeles]=5445

if [[ "$GREEDY" == "true" ]]; then greedy_glob="*greedyTrue*.jsonl"
else greedy_glob="*greedyFalse*.jsonl"; fi

# Format: "step_filename|experiment_config"
declare -A RUNS

##############################################################################
# MODELS TRAINED ON TIMIT
RUNS[seg_pxeus_frac0_0053]="step_001000.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_053]="step_000875.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_106]="step_001500.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_2]="step_002250.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_4]="step_000500.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_6]="step_000500.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac0_8]="step_000625.ckpt|inference/segmentation_pxeus"
RUNS[seg_pxeus_frac1_0]="step_000625.ckpt|inference/segmentation_pxeus"

RUNS[seg_xeuspr_frac0_0053]="step_000875.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_053]="step_000875.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_106]="step_000875.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_2]="step_000750.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_4]="step_000750.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_6]="step_000750.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac0_8]="step_001000.ckpt|inference/segmentation_xeuspr"
RUNS[seg_xeuspr_frac1_0]="step_001375.ckpt|inference/segmentation_xeuspr"
##############################################################################

# MODELS TRAINED ON BUCKEYE
RUNS[seg_buckeye_pxeus_frac0_0008]="step_001625.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_0076]="step_001250.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_0153]="step_001500.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_0306]="step_002000.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_0612]="step_002875.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_1223]="step_004500.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_2446]="step_004625.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_4893]="step_001000.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac0_9786]="step_001375.ckpt|inference/segmentation_pxeus"
RUNS[seg_buckeye_pxeus_frac1_0]="step_001750.ckpt|inference/segmentation_pxeus"

RUNS[seg_buckeye_frac0_0008]="step_004625.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_0076]="step_001500.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_0153]="step_001250.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_0306]="step_003125.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_0612]="step_004000.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_1223]="step_000875.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_2446]="step_001000.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_4893]="step_001375.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac0_9786]="step_001625.ckpt|inference/segmentation_xeuspr"
RUNS[seg_buckeye_frac1_0]="step_001500.ckpt|inference/segmentation_xeuspr"
##############################################################################

declare -A inference_dataset
inference_dataset[timit]=changelinglab/timit-segment
inference_dataset[buckeye]=changelinglab/buckeye-segment
inference_dataset[voxangeles]=changelinglab/voxangeles-segment

for run_folder in "${!RUNS[@]}"; do
    IFS='|' read -r ckpt_file exp <<< "${RUNS[$run_folder]}"
    ckpt=${BASE}/${run_folder}/checkpoints/${ckpt_file}

    hf_repo=${inference_dataset[${DATASET}]}
    out_run_folder=${run_folder}.on${DATASET}

    out_dir="${INF_BASE}/${out_run_folder}"
    shards=("${out_dir}"/${greedy_glob})
    if [[ -e "${shards[0]}" ]]; then
        actual=$(wc -l < <(cat "${shards[@]}"))
        expected="${EXPECTED_SAMPLES[${DATASET}]}"
        if [[ "${actual}" -eq "${expected}" ]]; then
            echo "Skipping ${run_folder} (already complete: ${actual} lines)"
            continue
        fi
    fi

    run_folder=${out_run_folder}
    sbatch -p ghx4 -t 4:00:00 scripts/daixpr_inference.batch \
        experiment=${exp} \
        run_folder=${run_folder} \
        inference.inference_runner.ckpt_path="${ckpt}" \
        inference.inference_runner.greedy=${GREEDY} \
        data.hf_repo="${hf_repo}" "$@"
done
