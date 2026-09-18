#!/bin/bash
# Full SSNCE (Tamil) forced alignment under NATIVE MFA 2.2.17, end to end:
#   1. PhoneticXEUS recognition (GPU, distributed_inference) -> predicted phones
#   2. mfa2 `mfa align`: cascade (predicted phones) + topline (gold phones)
#   3. score both with eval_segmentation
# tamil_cv is an MFA v2.0.0 model and mis-aligns under the project's MFA 3.3.9;
# the mfa2 env aligns it natively. Step 1 needs a GPU; the MFA align + scoring
# are CPU on the same node.
#
# Submit:  sbatch scripts/run_mfa2_tamil.sh
#
# Requires XEUSPR_CKPT=<path to the PhoneticXEUS fine-tuned .ckpt> in the
# environment (sbatch exports the submitting shell's environment).
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J mfa2_tamil
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1:30:00
#SBATCH -o exp/slurm_logs/%j.out
#SBATCH -e exp/slurm_logs/%j.out

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs
source scripts/env.sh
export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1
PY=.venv/bin/python

OUT_ROOT="exp/runs/mfa2_tamil/${SLURM_JOB_ID:-local}"
RECOG_DIR="$OUT_ROOT/recog"
RECOG_GLOB="$RECOG_DIR/ssnce_xeuspr_tamil_phones.*.jsonl"
mkdir -p "$RECOG_DIR"  # hydra writes hydra.log into paths.output_dir
echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1) out_root=$OUT_ROOT"

echo "=================== [1] PhoneticXEUS recognition -> $RECOG_DIR ==================="
# Pin a single shard (non-array job) so distributed_inference processes the full
# dataset; num_workers (config) parallelizes on the one GPU.
env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
  $PY src/main.py experiment=recognize/xeuspr_tamil paths.output_dir="$RECOG_DIR"

for SRC in cascade gt; do
  echo "=================== [2] mfa2 align [$SRC] ==================="
  $PY -m src.model.mfa.mfa2_align --source "$SRC" \
    --out-root "$OUT_ROOT" --recog-jsonl "$RECOG_GLOB"

  echo "=================== [3] eval [$SRC] ==================="
  $PY scripts/eval_segmentation.py "$OUT_ROOT/$SRC/pred.jsonl" \
    --tolerance-ms 20 --mode strict --strip-outer-silences \
    --out-csv "$OUT_ROOT/$SRC/metrics.csv"
done

echo "=================== summary (precision,recall,f1,rval,over_seg) ==================="
for SRC in gt cascade; do
  echo "[$SRC] $(tail -n 1 "$OUT_ROOT/$SRC/metrics.csv" 2>/dev/null)"
done
