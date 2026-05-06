# ENGLISH

export HF_HOME=exp/cache/hf
REPO=(
    changelinglab/timit-segment
    changelinglab/buckeye-segment
    changelinglab/gtimit-l1simple-segment
    changelinglab/gtimit-l2simple-segment
    changelinglab/gtimit-l1tbnk-segment
    changelinglab/gtimit-l2tbnk-segment
    changelinglab/torgo-segment
)

# Inference
for repo in ${REPO[@]}; do
    dataname=$(basename "$repo")
    dataname=${dataname%%-segment}
    echo "Running MFA topline for $dataname (repo: $repo) ..."
    python -m src.model.mfa.inference_topline \
        --hf_repo "$repo" \
        --split test \
        --mfa_cache_dir exp/cache/mfa \
        --run_dir exp/runs/mfa/topline_"$dataname"
done

# Evaluation
for repo in ${REPO[@]}; do
    dataname=$(basename "$repo")
    dataname=${dataname%%-segment}
    echo "Evaluating MFA topline for $dataname (repo: $repo) ..."
    if [[ "$dataname" == "timit" ]]; then
        extra_args="--strip-outer-silences"
    else
        extra_args=""
    fi
    python -m scripts.eval_segmentation "exp/runs/mfa/topline_${dataname}/results.jsonl" ${extra_args}
done