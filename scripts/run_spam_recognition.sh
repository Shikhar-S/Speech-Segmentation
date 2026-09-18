#!/bin/bash
# Phonvec recognition: distributed inference + PER/PFER eval across 6
# Kaldi-format datasets (TIMIT, l2arctic, GMU SAA, DoReCo, VoxAngeles,
# TUSOM 2021).
#
# Submit (all 6 datasets in parallel):
#   sbatch scripts/run_spam_recognition.sh
# Resubmit a single dataset (index into ORDER below; 0=timit … 5=tusom2021):
#   sbatch --array=4 scripts/run_spam_recognition.sh
# Override the encoder via first positional arg:
#   sbatch scripts/run_spam_recognition.sh juice500/wavlm-12-phonemodel
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J spam_recognition
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 4:00:00
#SBATCH --array=0-5
#SBATCH -o exp/slurm_logs/%A_%a.out
#SBATCH -e exp/slurm_logs/%A_%a.out


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

source scripts/env.sh

export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export TRANSFORMERS_VERBOSITY=warning
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


MODEL="${1:-juice500/wavlm-24-phonemodel}"
# All array tasks share ONE root via SLURM_ARRAY_JOB_ID, this 
# lets the per-task outputs be combined under a single run dir.
OUT_ROOT="exp/runs/phonvec_recognition/${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"

# Kaldi-format dataset names as keyed in configs/data/powsm_evalset_index.yaml;
# each has a configs/experiment/recognize/phonvec_<name>.yaml config.
ORDER=(timit l2arctic_perceived gmuaccent doreco voxangeles tusom2021)
NAME=${ORDER[${SLURM_ARRAY_TASK_ID:-0}]}
if [[ -z "$NAME" ]]; then
  echo "[error] no dataset at array index ${SLURM_ARRAY_TASK_ID:-0} (have ${#ORDER[@]})"
  exit 1
fi

ODIR="$OUT_ROOT/$NAME"
mkdir -p "$ODIR"
# Per-task summary row (one headerless line). Array tasks run in parallel, so a
# single shared CSV would race; each task owns its own file and the combined
# summary.csv is rebuilt from them.
ROW="$OUT_ROOT/summary.$NAME.csv"

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] model=$MODEL  array_task=${SLURM_ARRAY_TASK_ID:-0}  dataset=$NAME  out_root=$OUT_ROOT"

echo "=================== [$NAME] inference -> $ODIR ==================="
# distributed_inference.py shards the dataset by SLURM_ARRAY_TASK_ID /
# SLURM_ARRAY_TASK_COUNT. Here the array index selects a DATASET, not a shard,
# so pin id=0/count=1 to make each task process its FULL dataset as one shard
# (env wraps python to override srun's propagation of the real array vars).
srun --output="$ODIR/inference.log" --error="$ODIR/inference.log" \
  env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
  python src/main.py \
    experiment="recognize/phonvec_${NAME}" \
    inference.inference_runner.model_name_or_path="$MODEL" \
    inference.inference_runner.vocab_file=null \
    +inference.inference_runner.panphon_unrestricted=true \
    paths.output_dir="$ODIR"

echo "=================== [$NAME] postprocess (prism convention) ==================="
# Map model output phones to prism convention before scoring. seg2voxprism splits
# tie-less affricates for VoxAngeles; seg2prism (default) for the rest.
MAP="configs/inference/seg2prism_ipa_map.json"
[[ "$NAME" == "voxangeles" ]] && MAP="configs/inference/seg2voxprism_ipa_map.json"
# TIMIT's prism dump is the full corpus (6300); restrict to the held-out test
# split (1680, changelinglab/timit-segment test) before scoring.
KEEP=(); [[ "$NAME" == "timit" ]] && KEEP=(--keep-ids configs/inference/timit_test_ids.txt)
for raw in "$ODIR/${NAME}_phonvec".*.jsonl; do
  [[ -e "$raw" ]] || continue
  shard="${raw##*phonvec.}"
  python scripts/postprocess_for_prism.py "$raw" "$ODIR/${NAME}_phonvec_postproc.${shard}" "$MAP" "${KEEP[@]}"
done

echo "=================== [$NAME] eval (postprocessed) ==================="
python scripts/eval_recognition.py \
  "$ODIR/${NAME}_phonvec_postproc."*.jsonl \
  --out-csv "$ODIR/metrics.csv" \
  2>&1 | tee "$ODIR/eval.log"

# metrics.csv: header "n_utts,per,pfer,macro_lang_per" + one data row; keep the
# first three columns for the combined table (dataset,n_utts,per,pfer).
if [[ -f "$ODIR/metrics.csv" ]]; then
  echo "$NAME,$(tail -n 1 "$ODIR/metrics.csv" | cut -d, -f1-3)" > "$ROW"
else
  echo "$NAME,MISSING,MISSING,MISSING" > "$ROW"
fi

# Rebuild the combined summary from whatever per-task rows exist, in ORDER.
# Idempotent: every task does it, the last one to finish writes the full table.
# Write to a PID-unique temp then atomically mv, so concurrent rebuilds by
# sibling tasks can never produce a partially written summary.csv.
COMBINED="$OUT_ROOT/summary.csv"
TMP="$OUT_ROOT/.summary.$$.tmp"
{
  echo "dataset,n_utts,per,pfer"
  for n in "${ORDER[@]}"; do
    [[ -f "$OUT_ROOT/summary.$n.csv" ]] && cat "$OUT_ROOT/summary.$n.csv"
  done
} > "$TMP"
# Unconditional mv (not chained on the brace group's status — its last command
# is a per-dataset `[[ -f ]] && cat` that is false whenever the last dataset in
# ORDER has not finished, which must not block the rebuild).
mv -f "$TMP" "$COMBINED"

echo
echo "=================== [$NAME] done ==================="
cat "$ROW"
echo "(combined so far -> $COMBINED; complete once all ${#ORDER[@]} tasks finish)"
