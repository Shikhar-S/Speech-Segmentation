#!/bin/bash
# Phonvec recognition boundary ablation on the changelinglab/*-segment datasets
# (TIMIT, VoxAngeles). For each dataset, recognition runs twice on the SAME
# segment test set: oracle (GT phone_timestamps drive recognition) and pred
# (the model segmenter's predicted boundaries drive recognition, oracle=false). PER/PFER are scored
# against the GT `phones`
#
# Args (positional, Hydra-resolves last value):
#   1  MODEL     encoder (default juice500/wavlm-24-phonemodel)
#   2  REC_MODE  vocab (default) | panphon
#             vocab   : per-language panphon_featmap from each segment dataset's
#                       GT phones (vocab-constrained decode).
#             panphon : single Recognizer over the full PanPhon inventory
#                       (no featmap) — the paper's panphon-unrestricted oracle.
# Out dir: exp/runs/phonvec_oracle/<jobid>/<REC_MODE>/...
#
# SLURM array: one task per dataset (--array=0-1), each running both conditions.
#
# Submit (vocab):
#   sbatch scripts/run_phonvec_oracle.sh
# Submit (panphon-unrestricted):
#   sbatch scripts/run_phonvec_oracle.sh juice500/wavlm-24-phonemodel panphon
# Resubmit a single dataset (0=timit, 1=voxangeles), keeping REC_MODE:
#   sbatch --array=1 scripts/run_phonvec_oracle.sh juice500/wavlm-24-phonemodel panphon
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J phonvec_oracle
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --array=0-1
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
# Recognizer mode: vocab (per-language panphon_featmap, default) or panphon
# (single Recognizer over the full PanPhon inventory, no featmap — the paper's
# panphon-unrestricted oracle decode).
REC_MODE="${2:-vocab}"
REC_ARGS=()
if [[ "$REC_MODE" == "panphon" ]]; then
  REC_ARGS=(
    inference.inference_runner.vocab_file=null
    +inference.inference_runner.panphon_unrestricted=true
  )
fi
OUT_ROOT="exp/runs/phonvec_oracle/${SLURM_ARRAY_JOB_ID:-local}/$REC_MODE"
mkdir -p "$OUT_ROOT"

# name -> experiment config, HF segment repo, vocab JSON.
EXP_timit="recognize/phonvec_oracle_timit"
REPO_timit="${SEG_REPO_TIMIT:-changelinglab/timit-segment}"
VOCAB_timit="configs/inference/phonvec_vocab/timit_segment.json"

EXP_voxangeles="recognize/phonvec_oracle_voxangeles"
REPO_voxangeles="${SEG_REPO_VOXANGELES:-changelinglab/voxangeles-segment}"
VOCAB_voxangeles="configs/inference/phonvec_vocab/voxangeles_segment.json"

# One SLURM array task per dataset; each task runs BOTH boundary conditions on
# the SAME segment dataset.
ORDER=(timit voxangeles)
CONDS=(oracle pred)
NAME=${ORDER[$SLURM_ARRAY_TASK_ID]}
REPO="REPO_$NAME"; REPO=${!REPO}
VOCAB="VOCAB_$NAME"; VOCAB=${!VOCAB}
EXP="EXP_$NAME"; EXP=${!EXP}
# Per-task summary (array tasks run in parallel; a shared CSV would race).
SUMMARY="$OUT_ROOT/summary.$NAME.csv"
echo "dataset,cond,n_utts,per,pfer,macro_lang_per" > "$SUMMARY"

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] model=$MODEL  array_task=$SLURM_ARRAY_TASK_ID  dataset=$NAME  out_root=$OUT_ROOT"

if [[ ! -f "$VOCAB" ]]; then
  echo "=================== [$NAME] build vocab -> $VOCAB ==================="
  python scripts/build_phonvec_vocab_segment.py "$REPO" "$VOCAB" \
    2>&1 | tee "$OUT_ROOT/$NAME.vocab.log"
fi

run_one() {
  local cond=$1
  local oracle="true"; [[ "$cond" == "pred" ]] && oracle="false"
  local odir="$OUT_ROOT/$NAME/$cond"
  mkdir -p "$odir"

  echo
  echo "=========== [$NAME/$cond] inference (oracle=$oracle) -> $odir ==========="
  # distributed_inference.py shards by SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT.
  # Here the array index selects a DATASET, not a shard, so pin id=0/count=1 to
  # make each task process its full dataset as one shard (env wraps python to
  # override srun's propagation of the real array vars).
  srun --output="$odir/inference.log" --error="$odir/inference.log" \
    env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
    python src/main.py \
      experiment="$EXP" \
      inference.inference_runner.model_name_or_path="$MODEL" \
      inference.inference_runner.oracle="$oracle" \
      paths.output_dir="$odir" \
      "${REC_ARGS[@]}"

  echo "=================== [$NAME/$cond] eval ==================="
  python scripts/eval_recognition.py \
    "$odir/phonvec_oracle*.jsonl" \
    --out-csv "$odir/metrics.csv" \
    2>&1 | tee "$odir/eval.log"

  if [[ -f "$odir/metrics.csv" ]]; then
    echo "$NAME,$cond,$(tail -n 1 "$odir/metrics.csv")" >> "$SUMMARY"
  else
    echo "$NAME,$cond,MISSING,MISSING,MISSING,MISSING" >> "$SUMMARY"
  fi
}

for cond in "${CONDS[@]}"; do
  run_one "$cond"
done

echo
echo "=================== [$NAME] summary ==================="
cat "$SUMMARY"
echo "(aggregate across datasets: cat $OUT_ROOT/summary.*.csv)"

echo
echo "=================== summary ==================="
cat "$SUMMARY"
