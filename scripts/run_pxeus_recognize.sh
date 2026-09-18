#!/bin/bash
# Stage 1 of the decoupled English cascade: PhoneticXEUS masked recognition only.
# Dumps predicted phones per utterance.
#
# Submit:
#   sbatch scripts/run_pxeus_recognize.sh timit
#   sbatch scripts/run_pxeus_recognize.sh buckeye
#
# Requires XEUSPR_CKPT=<path to the PhoneticXEUS fine-tuned .ckpt> in the
# environment (sbatch exports the submitting shell's environment).
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J pxeus_recog
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o exp/slurm_logs/%j.out
#SBATCH -e exp/slurm_logs/%j.out

# Each distributed_inference worker loads the FULL dataset into RAM + its own
# XEUS copy, so num_workers(=4) * (dataset + XEUS) must fit. 32G OOM-kills a
# worker and hangs the pool; SSNCE (7860 utts, 4 workers) needs ~48G -> 64G here.

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

DATASET="${1:-timit}"
declare -A REPOS=(
  [timit]=${SEG_REPO_TIMIT:-changelinglab/timit-segment}
  [buckeye]=${SEG_REPO_BUCKEYE:-changelinglab/buckeye-segment}
  [gtimit_l2simple]=${SEG_REPO_GTIMIT_L2SIMPLE:-changelinglab/gtimit-l2simple-segment}
  [gtimit_l2tbnk]=${SEG_REPO_GTIMIT_L2TBNK:-changelinglab/gtimit-l2tbnk-segment}
  [torgo]=${SEG_REPO_TORGO:-changelinglab/torgo-segment}
)
REPO=${REPOS[$DATASET]}
if [[ -z "$REPO" ]]; then
  echo "Unknown dataset: $DATASET (valid: ${!REPOS[*]})" >&2; exit 2
fi
EXTRA=("${@:2}")

source scripts/env.sh
export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export HYDRA_FULL_ERROR=1
PY=.venv/bin/python

OUT_DIR="exp/runs/pxeus_recognize/${SLURM_JOB_ID:-local}/$DATASET"
mkdir -p "$OUT_DIR"
echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1) dataset=$DATASET out=$OUT_DIR"

# Pin a single shard (non-array) so distributed_inference covers the full set;
# num_workers (config) parallelizes on the one GPU.
env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
  $PY src/main.py experiment=recognize/xeuspr \
  data.hf_repo="$REPO" \
  paths.output_dir="$OUT_DIR" \
  "${EXTRA[@]}"

echo "[done] dump: $OUT_DIR/xeuspr_phones.0.jsonl ($(wc -l < "$OUT_DIR/xeuspr_phones.0.jsonl" 2>/dev/null) lines)"
