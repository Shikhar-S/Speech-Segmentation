#!/bin/bash
# SSL-model x layer ablation for phonvec SEGMENTATION.
#
# Ablation on four SSL encoders crossed
# with nine tap layers -> 36 juice500 PhoneModel repos. Each is the phonvec
# segmenter built on that encoder's features at that layer:
#   models : xls-r  w2v2  hubert  wavlm
#   layers : 0 3 6 9 12 15 18 21 24
#   repo   : juice500/<model>-<layer>-phonemodel   (wavlm-24 = the full model)
#
# One SLURM array task per (model, layer) variant; each runs inference + boundary
# eval on TIMIT and VoxAngeles.
#
# Submit (all 36 variants in parallel):
#   sbatch scripts/run_sslwlayer_ablation.sh
# Resubmit one variant (array index 0..35, row-major over models then layers;
# 0=xls-r-0, 8=xls-r-24, 9=w2v2-0, ... 35=wavlm-24):
#   sbatch --array=35 scripts/run_sslwlayer_ablation.sh
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J sslwlayer_ablation
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 1:00:00
#SBATCH --array=0-35
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
# array), so the per-variant outputs combine under a single run dir.
OUT_ROOT="exp/runs/sslwlayer_ablation/${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"

# Variants = SSL model x tap layer, row-major (model outer, layer inner). The
# slug doubles as the repo stem: juice500/<slug>-phonemodel.
MODELS=(xls-r w2v2 hubert wavlm)
LAYERS=(0 3 6 9 12 15 18 21 24)
VARIANTS=()
for m in "${MODELS[@]}"; do
  for l in "${LAYERS[@]}"; do
    VARIANTS+=("$m-$l")
  done
done

declare -A REPOS=(
  [timit]=${SEG_REPO_TIMIT:-changelinglab/timit-segment}
  [voxangeles]=${SEG_REPO_VOXANGELES:-changelinglab/voxangeles-segment}
)
DS_ORDER=(timit voxangeles)

SLUG=${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}
if [[ -z "$SLUG" ]]; then
  echo "[error] no variant at array index ${SLURM_ARRAY_TASK_ID:-0} (have ${#VARIANTS[@]})"
  exit 1
fi
MODEL="juice500/${SLUG}-phonemodel"

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] model=$MODEL  array_task=${SLURM_ARRAY_TASK_ID:-0}  out_root=$OUT_ROOT"

# Per-variant summary rows (one line per dataset). Array tasks run in parallel,
# so each variant owns its own file; the combined summary.csv is rebuilt from them.
ROW="$OUT_ROOT/summary.$SLUG.csv"
: > "$ROW"
for NAME in "${DS_ORDER[@]}"; do
  REPO=${REPOS[$NAME]}
  ODIR="$OUT_ROOT/$SLUG/$NAME"
  mkdir -p "$ODIR"

  echo "=================== [$SLUG/$NAME] inference -> $ODIR ==================="
  # distributed_inference.py shards by SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT;
  # the array index selects a VARIANT, not a shard, so pin id=0/count=1 to process
  # the full dataset as one shard (env overrides srun's propagation of the real vars).
  srun --output="$ODIR/inference.log" --error="$ODIR/inference.log" \
    env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
    python src/main.py \
      experiment=segment/phonvec \
      inference.inference_runner.model_name_or_path="$MODEL" \
      data.hf_repo="$REPO" \
      data.predict_split=test \
      paths.output_dir="$ODIR"

  echo "=================== [$SLUG/$NAME] eval ==================="
  python scripts/eval_segmentation.py \
    "$ODIR/phonvec*.jsonl" \
    --tolerance-ms 20 \
    --mode strict \
    --strip-outer-silences \
    --out-csv "$ODIR/metrics.csv" \
    2>&1 | tee "$ODIR/eval.log"

  if [[ -f "$ODIR/metrics.csv" ]]; then
    echo "$SLUG,$NAME,$(tail -n 1 "$ODIR/metrics.csv")" >> "$ROW"
  else
    echo "$SLUG,$NAME,MISSING,MISSING,MISSING,MISSING,MISSING" >> "$ROW"
  fi
done

# Rebuild the combined summary from whatever per-variant rows exist, in order.
# Idempotent; write to a PID-unique temp then atomic mv so concurrent rebuilds
# by sibling tasks never produce a partial summary.csv.
COMBINED="$OUT_ROOT/summary.csv"
TMP="$OUT_ROOT/.summary.$$.tmp"
{
  echo "model,dataset,precision,recall,f1,rval,over_seg"
  for v in "${VARIANTS[@]}"; do
    [[ -f "$OUT_ROOT/summary.$v.csv" ]] && cat "$OUT_ROOT/summary.$v.csv"
  done
} > "$TMP"
mv -f "$TMP" "$COMBINED"

echo
echo "=================== [$SLUG] done ==================="
cat "$ROW"
echo "(combined so far -> $COMBINED; complete once all ${#VARIANTS[@]} variants finish)"
