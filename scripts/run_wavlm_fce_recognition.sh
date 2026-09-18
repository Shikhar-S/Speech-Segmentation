#!/bin/bash
# WavLM-FCE recognition (SLURM array, 1 task per dataset).
# Distributed inference + prism-convention postprocess + PER/PFER eval across 6
# Kaldi-format datasets.
# FCESegmentationHead.decode emits per-frame argmax → phone-string units;
# postprocess_for_prism.py maps the training alphabet to prism convention (same
# step as SPAM/CTC); scripts/eval_recognition.py reads them via the head-keyed
# dict branch of parse_pred_labels (--head fa) and drops vocab reserved tokens.
#
# Submit:
#   sbatch scripts/run_wavlm_fce_recognition.sh <ckpt_path>
# Resubmit a single failed dataset:
#   sbatch --array=2 scripts/run_wavlm_fce_recognition.sh <ckpt_path>
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J wavlm_fce_recognition
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 0:45:00
#SBATCH --array=0-5
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

ORDER=(timit l2arctic_perceived gmuaccent doreco voxangeles tusom2021)
NAME=${ORDER[$SLURM_ARRAY_TASK_ID]}

OUT_ROOT="exp/runs/wavlm_fce_recognition/${SLURM_ARRAY_JOB_ID:-local}"
ODIR="$OUT_ROOT/$NAME"
mkdir -p "$ODIR"

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] ckpt=$CKPT_PATH  array_task=$SLURM_ARRAY_TASK_ID  dataset=$NAME"
echo "[setup] odir=$ODIR"

echo "=================== [$NAME] inference -> $ODIR ==================="
# distributed_inference.py shards the dataset by SLURM_ARRAY_TASK_ID /
# SLURM_ARRAY_TASK_COUNT. Here the array index selects a DATASET, not a shard,
# so pin id=0/count=1 to make each task process the full dataset as one shard
# (env wraps python to override srun propagation).
srun --output="$ODIR/inference.log" --error="$ODIR/inference.log" \
  env SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
  python src/main.py \
    experiment=recognize/wavlm_fce \
    data.dataset_name="$NAME" \
    inference.num_workers=2 \
    inference.inference_runner.ckpt_path="$CKPT_PATH" \
    paths.output_dir="$ODIR" \
    "$@"

echo "=================== [$NAME] postprocess (prism convention) ==================="
# Map model output phones from the timit-segment training alphabet to prism
# convention before scoring (same step as run_spam_recognition.sh, so CTC/FCE/SPAM
# are scored identically). seg2voxprism splits tie-less affricates for VoxAngeles;
# seg2prism (default) for the rest.
MAP="configs/inference/seg2prism_ipa_map.json"
[[ "$NAME" == "voxangeles" ]] && MAP="configs/inference/seg2voxprism_ipa_map.json"
# TIMIT's prism dump is the full corpus (6300); restrict to the held-out test
# split (1680, changelinglab/timit-segment test) before scoring.
KEEP=(); [[ "$NAME" == "timit" ]] && KEEP=(--keep-ids configs/inference/timit_test_ids.txt)
for raw in "$ODIR/${NAME}_wavlm_fce".*.jsonl; do
  [[ -e "$raw" ]] || continue
  shard="${raw##*wavlm_fce.}"
  python scripts/postprocess_for_prism.py "$raw" "$ODIR/${NAME}_wavlm_fce_postproc.${shard}" "$MAP" "${KEEP[@]}"
done

echo "=================== [$NAME] eval (postprocessed) ==================="
python scripts/eval_recognition.py \
  "$ODIR/${NAME}_wavlm_fce_postproc."*.jsonl \
  --head fa \
  --out-csv "$ODIR/metrics.csv" \
  2>&1 | tee "$ODIR/eval.log"

echo "=================== [$NAME] done ==================="
