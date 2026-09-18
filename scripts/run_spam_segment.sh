#!/bin/bash
# Phonvec segmentation: distributed inference + boundary eval.
#
# Datasets: TIMIT, Buckeye, gTIMIT-l2simple, gTIMIT-l2tbnk, TORGO, SSNCE,
#           VoxAngeles, gTIMIT-Thai. One SLURM array task per dataset, so all
#           datasets run in parallel (was a sequential for-loop in a single job).
# Encoder : WavLM-large, layer 24  ->  juice500/wavlm-24-phonemodel.
#
# Submit (all 8 datasets in parallel):
#   sbatch scripts/run_spam_segment.sh
# Resubmit a single dataset (index into ORDER below; 0=timit, 1=buckeye,
# 2=gtimit_l2simple, 3=gtimit_l2tbnk, 4=torgo, 5=ssnce, 6=voxangeles,
# 7=gtimit_tha):
#   sbatch --array=4 scripts/run_spam_segment.sh
# Override the encoder via first positional arg:
#   sbatch scripts/run_spam_segment.sh juice500/wavlm-12-phonemodel
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J spam_segment
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 4:00:00
#SBATCH --array=0-7
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
# All array tasks share ONE root via SLURM_ARRAY_JOB_ID (constant across the
# array), not SLURM_JOB_ID (unique per task) — that is what lets the per-task
# outputs be combined under a single run dir.
OUT_ROOT="exp/runs/phonvec_segmentation/${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"

# dataset name -> HF repo id
declare -A REPOS=(
  [timit]=${SEG_REPO_TIMIT:-changelinglab/timit-segment}
  [buckeye]=${SEG_REPO_BUCKEYE:-changelinglab/buckeye-segment}
  [gtimit_l2simple]=${SEG_REPO_GTIMIT_L2SIMPLE:-changelinglab/gtimit-l2simple-segment}
  [gtimit_l2tbnk]=${SEG_REPO_GTIMIT_L2TBNK:-changelinglab/gtimit-l2tbnk-segment}
  [torgo]=${SEG_REPO_TORGO:-changelinglab/torgo-segment}
  [ssnce]=${SEG_REPO_SSNCE:-changelinglab/ssnce-segment}
  [voxangeles]=${SEG_REPO_VOXANGELES:-changelinglab/voxangeles-segment}
  [gtimit_tha]=${SEG_REPO_GTIMIT_THA:-changelinglab/gtimit-tha-segment}
)
ORDER=(timit buckeye gtimit_l2simple gtimit_l2tbnk torgo ssnce voxangeles gtimit_tha)
NAME=${ORDER[${SLURM_ARRAY_TASK_ID:-0}]}
if [[ -z "$NAME" ]]; then
  echo "[error] no dataset at array index ${SLURM_ARRAY_TASK_ID:-0} (have ${#ORDER[@]})"
  exit 1
fi
REPO=${REPOS[$NAME]}

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
    experiment=segment/phonvec \
    inference.inference_runner.model_name_or_path="$MODEL" \
    data.hf_repo="$REPO" \
    data.predict_split=test \
    paths.output_dir="$ODIR"

echo "=================== [$NAME] eval ==================="
python scripts/eval_segmentation.py \
  "$ODIR/phonvec*.jsonl" \
  --tolerance-ms 20 \
  --mode strict \
  --strip-outer-silences \
  --out-csv "$ODIR/metrics.csv" \
  2>&1 | tee "$ODIR/eval.log"

# metrics.csv: header "precision,recall,f1,rval,over_seg" + one data row; keep
# the whole 5-field data row for the combined table.
if [[ -f "$ODIR/metrics.csv" ]]; then
  echo "$NAME,$(tail -n 1 "$ODIR/metrics.csv")" > "$ROW"
else
  echo "$NAME,MISSING,MISSING,MISSING,MISSING,MISSING" > "$ROW"
fi

# Rebuild the combined summary from whatever per-task rows exist, in ORDER.
# Idempotent: every task does it, the last one to finish writes the full table.
# Write to a PID-unique temp then atomically mv, so concurrent rebuilds by
# sibling tasks can never produce a partially written summary.csv.
COMBINED="$OUT_ROOT/summary.csv"
TMP="$OUT_ROOT/.summary.$$.tmp"
{
  echo "dataset,precision,recall,f1,rval,over_seg"
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
