#!/bin/bash
# WavLM-BCE segmentation (SLURM array, 1 task per dataset).
# Distributed segmentation inference + boundary eval.
#
# Submit:
#   sbatch scripts/run_bce_segment.sh <ckpt_path>
# Resubmit a single failed dataset:
#   sbatch --array=3 scripts/run_bce_segment.sh <ckpt_path>
# Custom subset:
#   sbatch --array=0,4,7 scripts/run_bce_segment.sh <ckpt_path>
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J bce_segment
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --array=0-7
#SBATCH -o exp/slurm_logs/%A_%a.out
#SBATCH -e exp/slurm_logs/%A_%a.out


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <ckpt_path> [extra hydra overrides...]" >&2
  exit 2
fi
CKPT_PATH="$1"
shift

source scripts/env.sh

export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export TRANSFORMERS_VERBOSITY=warning
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ORDER=(timit buckeye gtimit_l2simple gtimit_l2tbnk torgo ssnce voxangeles gtimit_tha)
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

NAME=${ORDER[$SLURM_ARRAY_TASK_ID]}
REPO=${REPOS[$NAME]}

OUT_ROOT="exp/runs/bce_segmentation/${SLURM_ARRAY_JOB_ID:-local}"
ODIR="$OUT_ROOT/$NAME"
mkdir -p "$ODIR"

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] ckpt=$CKPT_PATH  array_task=$SLURM_ARRAY_TASK_ID  dataset=$NAME  repo=$REPO"
echo "[setup] odir=$ODIR"

echo "=================== [$NAME] BCE inference -> $ODIR ==================="
# distributed_inference.py shards the dataset by SLURM_ARRAY_TASK_ID /
# SLURM_ARRAY_TASK_COUNT. Here the array index selects a DATASET, not a shard,
# so pin id=0/count=1 to make each task process the full dataset as one shard
# (env wraps python to override srun propagation).
srun --output="$ODIR/inference.log" --error="$ODIR/inference.log" \
  env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
  python src/main.py \
    experiment=segment/wavlm_bce \
    inference.inference_runner.ckpt_path="$CKPT_PATH" \
    inference.num_workers=2 \
    data.prediction_dataset.hf_repo="$REPO" \
    paths.output_dir="$ODIR" \
    "$@"

echo "=================== [$NAME] eval ==================="
python scripts/eval_segmentation.py \
  "$ODIR/wavlm_bce_16k*.jsonl" \
  --tolerance-ms 20 \
  --mode strict \
  --strip-outer-silences \
  --out-csv "$ODIR/metrics.csv" \
  2>&1 | tee "$ODIR/eval.log"

echo "=================== [$NAME] done ==================="
