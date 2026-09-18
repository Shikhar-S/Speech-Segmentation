#!/bin/bash
# MFA text-independent baseline: the Koel phone recognizer transcribes each
# utterance, then MFA forced-aligns the predicted transcript. Per-utterance
# alignment. (The PhoneticXEUS cascade now uses the decoupled two-stage path:
# recognize/xeuspr*.yaml -> mfa_align_dump [English] / mfa2_align [Tamil].)
#
# Arg 1 = dataset filter (empty = the 5 English sets). Outputs go to
# exp/runs/mfa_koel/<jobid>/.
#
# Submit:
#   sbatch scripts/run_koel_mfa_baseline.sh              # 5 English sets
#   sbatch scripts/run_koel_mfa_baseline.sh timit        # one dataset
#
#SBATCH -A bbjs-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4
#SBATCH -J mfa_baseline
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 5:00:00
#SBATCH -o exp/slurm_logs/%j.out
#SBATCH -e exp/slurm_logs/%j.out


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

DATASET_FILTER="${1:-}"
RECOGNIZER=koel
declare -A EXP=(
  [koel]=segment/mfa_baseline
)
declare -A GLOB=(
  [koel]="baseline_mfa_en*.jsonl"
)
# Args after <dataset> are forwarded verbatim as hydra overrides
# (e.g. inference.limit_samples=8 for a smoke test).
EXTRA=("${@:2}")

# MFA + Koel live in a separate micromamba env (Kaldi/MFA toolchain).
# Prepend the env lib so its libstdc++ wins over /lib64 inside micromamba run.
source scripts/env.sh
export LD_LIBRARY_PATH="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/mfa310/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$PWD/exp/cache/mfa}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

OUT_ROOT="exp/runs/mfa_${RECOGNIZER}/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "dataset,precision,recall,f1,rval,over_seg" > "$SUMMARY"

echo "[setup] node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[setup] out_root=$OUT_ROOT"

declare -A REPOS=(
  [timit]=${SEG_REPO_TIMIT:-changelinglab/timit-segment}
  [buckeye]=${SEG_REPO_BUCKEYE:-changelinglab/buckeye-segment}
  [gtimit_l2simple]=${SEG_REPO_GTIMIT_L2SIMPLE:-changelinglab/gtimit-l2simple-segment}
  [gtimit_l2tbnk]=${SEG_REPO_GTIMIT_L2TBNK:-changelinglab/gtimit-l2tbnk-segment}
  [torgo]=${SEG_REPO_TORGO:-changelinglab/torgo-segment}
)
# Default sweep is the 5 English sets (english_mfa). Koel is English-only;
# Tamil (ssnce) goes through the mfa2_align corpus path, not this script.
ORDER=(timit buckeye gtimit_l2simple gtimit_l2tbnk torgo)
if [[ -n "$DATASET_FILTER" ]]; then
  if [[ -z "${REPOS[$DATASET_FILTER]:-}" ]]; then
    echo "Unknown dataset: $DATASET_FILTER (valid: ${ORDER[*]})" >&2; exit 2
  fi
  ORDER=("$DATASET_FILTER")
fi

run_one() {
  local name=$1
  local repo=${REPOS[$name]}
  local odir="$OUT_ROOT/$name"
  mkdir -p "$odir"

  echo
  echo "=================== [$name] baseline inference -> $odir ==================="
  micromamba run -n mfa310 python src/main.py \
    experiment=${EXP[$RECOGNIZER]} \
    data.hf_repo="$repo" \
    data.predict_split=test \
    paths.output_dir="$odir" \
    "${EXTRA[@]}" \
    2>&1 | tee "$odir/inference.log"

  echo "=================== [$name] eval ==================="
  python scripts/eval_segmentation.py \
    "$odir/${GLOB[$RECOGNIZER]}" \
    --tolerance-ms 20 \
    --mode strict \
    --strip-outer-silences \
    --out-csv "$odir/metrics.csv" \
    2>&1 | tee "$odir/eval.log"

  if [[ -f "$odir/metrics.csv" ]]; then
    local row
    row=$(tail -n 1 "$odir/metrics.csv")
    echo "$name,$row" >> "$SUMMARY"
  else
    echo "$name,MISSING,MISSING,MISSING,MISSING,MISSING" >> "$SUMMARY"
  fi
}

for name in "${ORDER[@]}"; do
  run_one "$name"
done

echo
echo "=================== summary (baseline) ==================="
cat "$SUMMARY"
