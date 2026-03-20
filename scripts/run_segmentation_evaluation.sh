#!/bin/bash

TRAIN_DATASET=all          # timit | buckeye | all
INFER_DATASET=all          # timit | buckeye | voxangeles | all
GREEDY=true
BASE=exp/runs/inf_segmentation

declare -A EXPECTED_SAMPLES
EXPECTED_SAMPLES[timit]=6300
EXPECTED_SAMPLES[buckeye]=10477
EXPECTED_SAMPLES[voxangeles]=5445

SUMMARY_CSV="${BASE}/summary.csv"

discovered_runs=()

for dir in "${BASE}"/*/; do
    [[ -d "${dir}" ]] || continue
    run_folder=$(basename "${dir}")

    # Extract infer_ds from .on<dataset> suffix; default to timit (legacy)
    if [[ "${run_folder}" =~ \.on([^.]+)$ ]]; then
        infer_ds="${BASH_REMATCH[1]}"
    else
        infer_ds=timit
    fi

    # Extract train_ds: buckeye if name contains "buckeye" before the .on suffix
    name_before_on="${run_folder%%.*}"
    if [[ "${name_before_on}" == *buckeye* ]]; then
        train_ds=buckeye
    else
        train_ds=timit
    fi

    [[ "${TRAIN_DATASET}" != "all" && "${train_ds}" != "${TRAIN_DATASET}" ]] && continue
    [[ "${INFER_DATASET}" != "all" && "${infer_ds}" != "${INFER_DATASET}" ]] && continue

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

    expected="${EXPECTED_SAMPLES[${infer_ds}]}"
    if [[ -n "${expected}" ]]; then
        actual=$(wc -l < <(cat "${shards[@]}"))
        if [[ "${actual}" -ne "${expected}" ]]; then
            echo "Skipping ${run_folder} (${actual}/${expected} lines, incomplete)"
            continue
        fi
    fi

    discovered_runs+=("${run_folder}:${train_ds}:${infer_ds}")

    metrics_csv="${dir}/metrics.csv"
    if [[ -f "${metrics_csv}" ]]; then
        echo "Skipping ${run_folder} (metrics.csv already exists)"
        continue
    fi

    echo "=== ${run_folder} train=${train_ds} infer=${infer_ds} forced=${forced} extra args (${@}) ==="
    echo "Shards: ${shards[*]}"
    args=("${shards[@]}")
    [[ "${forced}" == "true" ]] && args+=("--forced")
    args+=("--out-csv" "${metrics_csv}")
    python -m src.recipe.segmentation.local.eval_segmentation "${args[@]}" "$@"
done

# === SUMMARY CSV BLOCK ===
echo "run_folder,train_dataset,infer_dataset,ratio,rval,f1,precision,recall" > "${SUMMARY_CSV}"
for entry in "${discovered_runs[@]}"; do
    run_folder="${entry%%:*}"
    rest="${entry#*:}"
    train_ds="${rest%%:*}"
    infer_ds="${rest#*:}"
    metrics_csv="${BASE}/${run_folder}/metrics.csv"
    [[ -f "${metrics_csv}" ]] || continue
    python -c "
import csv, re, sys
with open('${metrics_csv}') as f:
    reader = csv.DictReader(f)
    row = next(reader, None)
if row is None:
    print('Warning: empty CSV for ${run_folder}, skipping', file=sys.stderr)
else:
    m = re.search(r'frac(\d+)_(\d+)', '${run_folder}')
    ratio = float(m.group(1) + '.' + m.group(2)) if m else float('nan')
    print('${run_folder},${train_ds},${infer_ds},' + f'{ratio},' + ','.join(f'{float(row[k]):.4f}' for k in ('rval','f1','precision','recall')))
" >> "${SUMMARY_CSV}"
done
echo "Summary written to ${SUMMARY_CSV}"
