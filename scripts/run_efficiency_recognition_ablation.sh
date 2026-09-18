#!/bin/bash
# Data-efficiency ablation for phonvec RECOGNITION (PER / PFER) on TIMIT + VoxAngeles.
#
# inference via recognize/phonvec_<ds> -> prism postprocessing -> eval_recognition (PER/PFER).
# Prism postprocessing is applied BEFORE scoring (seg2voxprism for VoxAngeles,
# seg2prism otherwise), matching run_spam_recognition.sh.
#
# Submit (all fractions in parallel):
#   sbatch scripts/run_efficiency_recognition_ablation.sh
# Resubmit one fraction (array index into FRACS; 0=1_1, 1=1_2, ... 10=1_1024):
#   sbatch --array=10 scripts/run_efficiency_recognition_ablation.sh
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J efficiency_recognition
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 2:00:00
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
OUT_ROOT="exp/runs/efficiency_recognition_ablation/${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"

FRACS=(1_1 1_2 1_4 1_8 1_16 1_32 1_64 1_128 1_256 1_512 1_1024)
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
  ODIR="$OUT_ROOT/$FRAC/$NAME"
  mkdir -p "$ODIR"

  echo "=================== [$FRAC/$NAME] inference -> $ODIR ==================="
  # The recognize/phonvec_<ds> config carries its own (Kaldi prism) data + vocab,
  # so no data override is needed — matches run_spam_recognition.sh. Pin the array
  # shard to 0/1 so each task processes the full dataset as one shard.
  srun --output="$ODIR/inference.log" --error="$ODIR/inference.log" \
    env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
    python src/main.py \
      experiment="recognize/phonvec_${NAME}" \
      inference.inference_runner.model_name_or_path="$MODEL" \
      paths.output_dir="$ODIR"

  echo "=================== [$FRAC/$NAME] postprocess (prism convention) ==================="
  # seg2voxprism splits tie-less affricates for VoxAngeles; seg2prism for the rest.
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

  echo "=================== [$FRAC/$NAME] eval (postprocessed) ==================="
  python scripts/eval_recognition.py \
    "$ODIR/${NAME}_phonvec_postproc."*.jsonl \
    --out-csv "$ODIR/metrics.csv" \
    2>&1 | tee "$ODIR/eval.log"

  # metrics.csv: header "n_utts,per,pfer,macro_lang_per" + one data row; keep the
  # first three columns for the combined table (frac,dataset,n_utts,per,pfer).
  if [[ -f "$ODIR/metrics.csv" ]]; then
    echo "$FRAC,$NAME,$(tail -n 1 "$ODIR/metrics.csv" | cut -d, -f1-3)" >> "$ROW"
  else
    echo "$FRAC,$NAME,MISSING,MISSING,MISSING" >> "$ROW"
  fi
done

# Rebuild the combined summary from whatever per-fraction rows exist, in order.
# Idempotent; write to a PID-unique temp then atomic mv so concurrent rebuilds
# by sibling tasks never produce a partial summary.csv.
COMBINED="$OUT_ROOT/summary.csv"
TMP="$OUT_ROOT/.summary.$$.tmp"
{
  echo "frac,dataset,n_utts,per,pfer"
  for f in "${FRACS[@]}"; do
    [[ -f "$OUT_ROOT/summary.$f.csv" ]] && cat "$OUT_ROOT/summary.$f.csv"
  done
} > "$TMP"
mv -f "$TMP" "$COMBINED"

echo
echo "=================== [$FRAC] done ==================="
cat "$ROW"
echo "(combined so far -> $COMBINED; complete once all ${#FRACS[@]} fractions finish)"
