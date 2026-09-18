#!/bin/bash
# MFA text-dependent topline: aligner sees ground-truth phones from the
# dataset (no recognizer in the loop). Best-case MFA boundary quality.
# Per-utterance alignment over TIMIT / Buckeye / gTIMIT-l2simple /
# gTIMIT-l2tbnk / TORGO (english_mfa), SSNCE Tamil (tamil_cv), and gTIMIT-Thai
# (thai_mfa). CPU-only (no GPU needed without the recognizer).
#
# Submit:
#   sbatch scripts/run_mfa_topline.sh                 # all 7 datasets, sequential
#   sbatch scripts/run_mfa_topline.sh ssnce           # single dataset (Tamil)
#   sbatch scripts/run_mfa_topline.sh gtimit_tha      # single dataset (Thai)
#
#SBATCH -A bbjs-delta-cpu
#SBATCH -p cpu
#SBATCH -J mfa_topline
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 5:00:00
#SBATCH -o exp/slurm_logs/%j.out
#SBATCH -e exp/slurm_logs/%j.out


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p exp/slurm_logs

DATASET_FILTER="${1:-}"

source scripts/env.sh
export LD_LIBRARY_PATH="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/mfa310/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$PWD/exp/cache/mfa}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

OUT_ROOT="exp/runs/mfa_topline/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "dataset,precision,recall,f1,rval,over_seg" > "$SUMMARY"

echo "[setup] node=$(hostname)"
echo "[setup] out_root=$OUT_ROOT"

declare -A REPOS=(
  [timit]=${SEG_REPO_TIMIT:-changelinglab/timit-segment}
  [buckeye]=${SEG_REPO_BUCKEYE:-changelinglab/buckeye-segment}
  [gtimit_l2simple]=${SEG_REPO_GTIMIT_L2SIMPLE:-changelinglab/gtimit-l2simple-segment}
  [gtimit_l2tbnk]=${SEG_REPO_GTIMIT_L2TBNK:-changelinglab/gtimit-l2tbnk-segment}
  [torgo]=${SEG_REPO_TORGO:-changelinglab/torgo-segment}
  [ssnce]=${SEG_REPO_SSNCE:-changelinglab/ssnce-segment}
  [gtimit_tha]=${SEG_REPO_GTIMIT_THA:-changelinglab/gtimit-tha-segment}
)
# Per-dataset topline experiment config. English datasets share english_mfa;
# SSNCE uses the Tamil tamil_cv model; gTIMIT-Thai uses thai_mfa
declare -A EXP=(
  [timit]=segment/mfa_topline
  [buckeye]=segment/mfa_topline
  [gtimit_l2simple]=segment/mfa_topline
  [gtimit_l2tbnk]=segment/mfa_topline
  [torgo]=segment/mfa_topline
  [ssnce]=segment/mfa_topline_tamil
  [gtimit_tha]=segment/mfa_topline_thai
)
ORDER=(timit buckeye gtimit_l2simple gtimit_l2tbnk torgo ssnce gtimit_tha)
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
  echo "=================== [$name] topline inference -> $odir ==================="
  micromamba run -n mfa310 python src/main.py \
    experiment="${EXP[$name]}" \
    data.hf_repo="$repo" \
    data.predict_split=test \
    paths.output_dir="$odir" \
    2>&1 | tee "$odir/inference.log"

  echo "=================== [$name] eval ==================="
  # Glob matches both english (topline_mfa_en*) and tamil (topline_mfa_tamil*)
  # output names; each $odir holds only its own dataset's jsonl.
  python scripts/eval_segmentation.py \
    "$odir/topline_mfa_*.jsonl" \
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
echo "=================== summary (topline) ==================="
cat "$SUMMARY"
