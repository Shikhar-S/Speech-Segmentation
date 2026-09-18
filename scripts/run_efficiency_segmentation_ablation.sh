#!/bin/bash
# Data-efficiency ablation for phonvec SEGMENTATION.
#
# wavlm-24 PhoneModels trained on a decreasing fraction of the training data:
#   1_1 (full data = wavlm-24-phonemodel itself), 1_2, 1_4, 1_8, 1_16, 1_32,
#   1_64, 1_128, 1_256, 1_512, 1_1024.
# One SLURM array task per fraction; each runs inference + boundary eval on TIMIT
# and VoxAngeles.
#
# Submit (all fractions in parallel):
#   sbatch scripts/run_efficiency_segmentation_ablation.sh
# Resubmit one fraction (array index into FRACS below; 0=1_1, 1=1_2, ... 10=1_1024):
#   sbatch --array=10 scripts/run_efficiency_segmentation_ablation.sh
#
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J efficiency_ablation
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 1:00:00
#SBATCH --array=0-10
#SBATCH -o exp/slurm_logs/%A_%a.out
#SBATCH -e exp/slurm_logs/%A_%a.out


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

source scripts/env.sh

export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export TRANSFORMERS_VERBOSITY=warning
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# All array tasks share ONE root via SLURM_ARRAY_JOB_ID (constant across the
# array), so the per-fraction outputs combine under a single run dir.
OUT_ROOT="exp/runs/efficiency_segmentation_ablation/${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"

FRACS=(1_1 1_2 1_4 1_8 1_16 1_32 1_64 1_128 1_256 1_512 1_1024)
declare -A REPOS=(
  [timit]=${SEG_REPO_TIMIT:-changelinglab/timit-segment}
  [voxangeles]=${SEG_REPO_VOXANGELES:-changelinglab/voxangeles-segment}
)
DS_ORDER=(timit voxangeles)

FRAC=${FRACS[${SLURM_ARRAY_TASK_ID:-0}]}
if [[ -z "$FRAC" ]]; then
  echo "[error] no fraction at array index ${SLURM_ARRAY_TASK_ID:-0} (have ${#FRACS[@]})"
  exit 1
fi
# 1_1 = full data = the base wavlm-24-phonemodel; the rest are the -efficiency- repos.
if [[ "$FRAC" == "1_1" ]]; then
  MODEL="juice500/wavlm-24-phonemodel"
else
  MODEL="juice500/wavlm-24-efficiency-${FRAC}-phonemodel"
fi

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] model=$MODEL  array_task=${SLURM_ARRAY_TASK_ID:-0}  frac=$FRAC  out_root=$OUT_ROOT"

# Per-fraction summary rows (one line per dataset). Array tasks run in parallel,
# so each fraction owns its own file; the combined summary.csv is rebuilt from them.
ROW="$OUT_ROOT/summary.$FRAC.csv"
: > "$ROW"
for NAME in "${DS_ORDER[@]}"; do
  REPO=${REPOS[$NAME]}
  ODIR="$OUT_ROOT/$FRAC/$NAME"
  mkdir -p "$ODIR"

  echo "=================== [$FRAC/$NAME] inference -> $ODIR ==================="
  # distributed_inference.py shards by SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT;
  # the array index selects a FRACTION, not a shard, so pin id=0/count=1 to process
  # the full dataset as one shard (env overrides srun's propagation of the real vars).
  srun --output="$ODIR/inference.log" --error="$ODIR/inference.log" \
    env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
    python src/main.py \
      experiment=segment/phonvec \
      inference.inference_runner.model_name_or_path="$MODEL" \
      data.hf_repo="$REPO" \
      data.predict_split=test \
      paths.output_dir="$ODIR"

  echo "=================== [$FRAC/$NAME] eval ==================="
  python scripts/eval_segmentation.py \
    "$ODIR/phonvec*.jsonl" \
    --tolerance-ms 20 \
    --mode strict \
    --strip-outer-silences \
    --out-csv "$ODIR/metrics.csv" \
    2>&1 | tee "$ODIR/eval.log"

  if [[ -f "$ODIR/metrics.csv" ]]; then
    echo "$FRAC,$NAME,$(tail -n 1 "$ODIR/metrics.csv")" >> "$ROW"
  else
    echo "$FRAC,$NAME,MISSING,MISSING,MISSING,MISSING,MISSING" >> "$ROW"
  fi
done

# Rebuild the combined summary from whatever per-fraction rows exist, in order.
# Idempotent; write to a PID-unique temp then atomic mv so concurrent rebuilds
# by sibling tasks never produce a partial summary.csv.
COMBINED="$OUT_ROOT/summary.csv"
TMP="$OUT_ROOT/.summary.$$.tmp"
{
  echo "frac,dataset,precision,recall,f1,rval,over_seg"
  for f in "${FRACS[@]}"; do
    [[ -f "$OUT_ROOT/summary.$f.csv" ]] && cat "$OUT_ROOT/summary.$f.csv"
  done
} > "$TMP"
mv -f "$TMP" "$COMBINED"

echo
echo "=================== [$FRAC] done ==================="
cat "$ROW"
echo "(combined so far -> $COMBINED; complete once all ${#FRACS[@]} fractions finish)"
