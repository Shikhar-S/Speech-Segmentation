#!/bin/bash

BASE=exp/runs/inf_segmentation
DUMP_CSV=false
GREEDY=true

RUNS=(
    # seg_pxeus_frac0_053
    # seg_pxeus_frac0_106
    # seg_pxeus_frac0_2
    # seg_pxeus_frac0_4
    # seg_pxeus_frac0_6
    # seg_pxeus_frac0_8
    # seg_pxeus_frac1_0

    seg_xeuspr_frac0_053
    seg_xeuspr_frac0_106
    seg_xeuspr_frac0_2
    seg_xeuspr_frac0_4
    seg_xeuspr_frac0_6
    seg_xeuspr_frac0_8
    seg_xeuspr_frac1_0
)

for run_folder in "${RUNS[@]}"; do
    dir="${BASE}/${run_folder}"
    if [[ "${GREEDY}" == "true" ]]; then
        shards=("${dir}"/*greedyTrue*.jsonl)
        forced=false
    else
        shards=("${dir}"/*greedyFalse*.jsonl)
        forced=true
    fi
    if [[ ! -e "${shards[0]}" ]]; then
        echo "Skipping ${run_folder} (no JSONL files found for GREEDY=${GREEDY})"
        continue
    fi

    echo "=== ${run_folder} (forced=${forced}) extra args (${@}) ==="
    echo "Shards: ${shards[*]}"
    args=("${shards[@]}")
    [[ "${forced}" == "true" ]] && args+=("--forced")
    [[ "${DUMP_CSV}" == "true" ]] && args+=("--out-csv" "${dir}/metrics.csv")
    python -m src.recipe.segmentation.local.eval_segmentation "${args[@]}" "$@"
done
