#!/bin/bash
# Stage 2 of the English MFA cascade: mfa align_one over the Stage-1
# recognition dump (scripts/run_pxeus_recognize.sh), then eval. No recognizer
# -> CPU only.
#
# Submit:  sbatch scripts/run_mfa_align_dump.sh <dataset> <recog_glob>
#   e.g.   sbatch scripts/run_mfa_align_dump.sh timit \
#            'exp/runs/pxeus_recognize/<jobid>/timit/xeuspr_phones.*.jsonl'
#
#SBATCH -A bbjs-delta-cpu
#SBATCH -p cpu
#SBATCH -J mfa_align_dump
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 5:00:00
#SBATCH -o exp/slurm_logs/%j.out
#SBATCH -e exp/slurm_logs/%j.out

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

DATASET="${1:?usage: run_mfa_align_dump.sh <dataset> <recog_glob>}"
RECOG_GLOB="${2:?usage: run_mfa_align_dump.sh <dataset> <recog_glob>}"
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

source scripts/env.sh
export LD_LIBRARY_PATH="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/mfa310/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$PWD/exp/cache/mfa}"
export HYDRA_FULL_ERROR=1

OUT_DIR="exp/runs/mfa_align_dump/${SLURM_JOB_ID:-local}/$DATASET"
mkdir -p "$OUT_DIR"
echo "[setup] node=$(hostname) dataset=$DATASET recog=$RECOG_GLOB out=$OUT_DIR"

env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
  micromamba run -n mfa310 python src/main.py \
  experiment=inference/mfa_align_dump \
  data.hf_repo="$REPO" \
  inference.inference_runner.recog_jsonl="$RECOG_GLOB" \
  paths.output_dir="$OUT_DIR"

echo "=================== eval ==================="
.venv/bin/python scripts/eval_segmentation.py \
  "$OUT_DIR/mfa_align_dump.*.jsonl" \
  --tolerance-ms 20 --mode strict --strip-outer-silences \
  --out-csv "$OUT_DIR/metrics.csv"
echo "[$DATASET] $(tail -n 1 "$OUT_DIR/metrics.csv" 2>/dev/null)"
