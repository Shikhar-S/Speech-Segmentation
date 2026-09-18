#!/bin/bash
# Phonvec recognition WITHOUT per-language vocab routing ("vocabulary = none"):
# distributed inference + PER/PFER eval on TIMIT and VoxAngeles.
#
# Uses the base recognize/phonvec config (vocab_file: null), so each utterance
# is decoded with the model's own IPA-view featmap recognizer instead of a
# per-language panphon vocab. Contrast with scripts/run_spam_recognition.sh,
# which routes through the precomputed per-language vocab JSONs.
#
# Submit:
#   sbatch scripts/run_spam_recognition_novocab.sh
# Override the encoder via first positional arg:
#   sbatch scripts/run_spam_recognition_novocab.sh juice500/wavlm-12-phonemodel
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J spam_recognition_novocab
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH -t 2:00:00
#SBATCH -o exp/slurm_logs/%j.out
#SBATCH -e exp/slurm_logs/%j.out


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

source scripts/env.sh

export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export TRANSFORMERS_VERBOSITY=warning
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


MODEL="${1:-juice500/wavlm-24-phonemodel}"
OUT_ROOT="exp/runs/phonvec_recognition_novocab/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "dataset,n_utts,per,pfer" > "$SUMMARY"

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] model=$MODEL  out_root=$OUT_ROOT  vocab=none"

# Kaldi-format dataset names as keyed in configs/data/powsm_evalset_index.yaml.
ORDER=(timit voxangeles)

run_one() {
  local name=$1
  local odir="$OUT_ROOT/$name"
  mkdir -p "$odir"

  echo
  echo "=================== [$name] inference -> $odir ==================="
  # Base recognize/phonvec config keeps vocab_file: null; we only pin the
  # dataset and override the encoder. No per-language routing.
  srun --output="$odir/inference.log" --error="$odir/inference.log" \
    python src/main.py \
      experiment=recognize/phonvec \
      data.dataset_name="$name" \
      inference.inference_runner.model_name_or_path="$MODEL" \
      paths.output_dir="$odir"

  echo "=================== [$name] postprocess (prism convention) ==================="
  # Mirror run_spam_recognition.sh so vocab/none are scored identically: prism
  # remap (no-op for phonvec) + TIMIT restricted to the held-out test-1680 split.
  MAP="configs/inference/seg2prism_ipa_map.json"
  [[ "$name" == "voxangeles" ]] && MAP="configs/inference/seg2voxprism_ipa_map.json"
  KEEP=(); [[ "$name" == "timit" ]] && KEEP=(--keep-ids configs/inference/timit_test_ids.txt)
  for raw in "$odir/${name}_phonvec".*.jsonl; do
    [[ -e "$raw" ]] || continue
    shard="${raw##*phonvec.}"
    python scripts/postprocess_for_prism.py "$raw" "$odir/${name}_phonvec_postproc.${shard}" "$MAP" "${KEEP[@]}"
  done

  echo "=================== [$name] eval (postprocessed) ==================="
  python scripts/eval_recognition.py \
    "$odir/${name}_phonvec_postproc."*.jsonl \
    --out-csv "$odir/metrics.csv" \
    2>&1 | tee "$odir/eval.log"

  if [[ -f "$odir/metrics.csv" ]]; then
    local row
    row=$(tail -n 1 "$odir/metrics.csv")
    echo "$name,$row" >> "$SUMMARY"
  else
    echo "$name,MISSING,MISSING,MISSING" >> "$SUMMARY"
  fi
}

for name in "${ORDER[@]}"; do
  run_one "$name"
done

echo
echo "=================== summary ==================="
cat "$SUMMARY"
