#!/usr/bin/env bash
# Copyright 2025  Carnegie Mellon University (Author: Shikhar Bharadwaj)
# TODO(shikhar): support for cascade setup
set -euo pipefail
log() {
    echo "$(date '+%Y-%m-%dT%H:%M:%S') [${BASH_SOURCE[1]##*/}:${BASH_LINENO[0]}] $*"
}

model="all"
recipe="all"
cluster="dai"
setup="probing"
fft=false
parallel=false
wait_time=10
dry_run=false
run_name=""
sbatch_args=""
extra_args=""

help_message=$(cat << 'EOF'
Usage: $0 [OPTIONS]

Options:
  --model LIST        Models: logmel, powsm, powsmvr, ctag, lv60, xlsr53, zipactc, zipactc_ns, or "all"
  --recipe LIST       Recipes: fab, fat, gsw, gva, l1c, l2a, lif, or "all"
  --cluster NAME      Cluster: dai, delta (default: dai)
  --setup NAME        Setup: probing, inference (default: probing)
  --fft               Enable full fine-tuning
  --parallel          Run in parallel
  --dry_run           Print commands only (explicitly set to --dry_run true)
  --run_name STR      Run name (default: timestamp)
  --sbatch_args STR   Extra sbatch arguments
  --extra_args STR    Extra training arguments
  
Examples:
  $0 --model powsm,ctag --recipe lif --fft
  $0 --model all --recipe fab,fat --cluster delta
  $0 --model ctag --recipe lif --setup inference
EOF
)
. scripts/parse_options.sh 2>/dev/null || true

flag_compatibility_checks(){
    if [[ "$setup" == "inference" ]]; then
        fft=false
        [[ -z "$extra_args" ]] && { echo "Set data field. Eg --extra_args data=fleurs"; }
    fi
}
flag_compatibility_checks

[ -z "$run_name" ] && run_name=$(date "+%Y%m%d%H%M%S")
exp_dir="$(pwd)/exp/runs"
mkdir -p "$exp_dir"
summary_log="${exp_dir}/${run_name}.summary.log"

# Cluster configurations
declare -A cluster_configs=(
    ["dai"]="scripts/dai.batch"
    ["delta"]="scripts/delta.batch"
)

# Model configurations: base_model|hf_repo
declare -A model_configs=(
    ["logmel"]="logmel|"
    ["powsm"]="powsm|"
    ["powsmvr"]="powsmvr|"
    ["ctag"]="w2v2ph|ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
    ["lv60"]="w2v2ph|facebook/wav2vec2-lv-60-espeak-cv-ft"
    ["xlsr53"]="w2v2ph|facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    ["zipactc"]="zipactc|anyspeech/zipa-large-crctc-500k"
    ["zipactc_ns"]="zipactc|anyspeech/zipa-large-crctc-ns-800k"
)

# Recipe = task_dataset
declare -A recipe_configs=(
    ["fab"]="fa_buckeye"
    ["fat"]="fa_timit"
    ["geo_sw"]="geolocation_swissgerman"
    ["geo_in"]="geolocation_vaani"
    ["l1cls"]="l1cls_cmul2arctic"
    ["l2as"]="l2as_speechocean"
    ["lid_fl"]="lid_fleurs"
    ["pr"]="transcribe"
)

get_base_model() {
    IFS='|' read -r base _ <<< "${model_configs[$1]}"
    echo "$base"
}

get_hf_repo() {
    IFS='|' read -r _ repo <<< "${model_configs[$1]}"
    echo "$repo"
}

config_exists() {
    local config_file=$1
    [ -f "$config_file" ]
}

generate_list() {
    local type=$1 input=$2
    local -n config_ref=$3
    local items=()
    
    if [ "$input" = "all" ]; then
        items=("${!config_ref[@]}")
    else
        IFS=',' read -ra requested <<< "$input"
        for item in "${requested[@]}"; do
            [[ -v config_ref[$item] ]] && items+=("$item")
        done
    fi
    
    [ ${#items[@]} -eq 0 ] && { log "No ${type}s found"; exit 1; }
    echo "${items[@]}"
}

construct_config_name() {
    local model_var=$1 recipe_code=$2
    local base=$(get_base_model "$model_var")
    local recipe_full="${recipe_configs[$recipe_code]}"
    echo "configs/experiment/${setup}/${recipe_full}_${base}.yaml"
}

construct_command() {
    local model_var=$1 config_file=$2
    local repo=$(get_hf_repo "$model_var")
    local script="${cluster_configs[$cluster]}"
    local cmd="sbatch${sbatch_args:+ $sbatch_args} $script experiment=${setup}/${config_file##*/}"
    
    if [ -n "$repo" ]; then
        case "$setup" in
            probing) cmd+=" model.net.hf_repo=$repo" ;;
            inference) cmd+=" inference.inference_runner.hf_repo=$repo" ;;
        esac
    fi
    
    if $fft; then
        cmd+=" model.freeze_encoder=false tags+=[\\\"fft\\\"]"
    fi
    
    [ -n "$extra_args" ] && cmd+=" $extra_args"
    echo "$cmd"
}

run_experiment() {
    local model_var=$1 recipe_code=$2
    local config_file=$(construct_config_name "$model_var" "$recipe_code")
    local cmd=$(construct_command "$model_var" "$config_file") || {
        log "Error constructing command for $model_var on $recipe_code (config: $config_file)"
        echo "ERROR: $model_var on $recipe_code" >> "$summary_log"
        return 1
    }
    if ! config_exists "$config_file"; then
        log "Skip: $model_var on $recipe_code (config not found: $config_file)"
        echo "SKIP: $model_var on $recipe_code" >> "$summary_log"
        return 2
    fi
    
    log "Run: $model_var on $recipe_code"
    echo "RUN: $model_var on $recipe_code" >> "$summary_log"
    log "CMD: $cmd"
    echo "CMD: $cmd" >> "$summary_log"
    
    [[ "$dry_run" = true ]] && return 0
    
    if $parallel; then
        eval "$cmd &"
        sleep "$wait_time"
    else
        if eval "$cmd"; then
            echo "SUCCESS: $model_var on $recipe_code" >> "$summary_log"
        else
            log "Failed: $model_var on $recipe_code"
            echo "FAILED: $model_var on $recipe_code" >> "$summary_log"
            return 1
        fi
    fi
}

# Initialize
{
    echo "=== Run: $run_name ==="
    echo "Started: $(date)"
    echo "Setup: $setup | Cluster: $cluster | FFT: $fft"
    echo ""
} > "$summary_log"

models=$(generate_list "model" "$model" model_configs)
recipes=$(generate_list "recipe" "$recipe" recipe_configs)

log "Models: $models"
log "Recipes: $recipes"

total=0 successful=0 failed=0 skipped=0

for m in $models; do
    for r in $recipes; do
        total=$((total + 1))
        set +e
        run_experiment "$m" "$r"
        set -e
        case $? in
            0) successful=$((successful + 1)) ;;
            2) skipped=$((skipped + 1)) ;;
            *) failed=$((failed + 1)) ;;
        esac
    done
done

$parallel && [[ "$dry_run" != true ]] && { log "Waiting for jobs..."; wait; }

{
    echo ""
    echo "=== Summary ==="
    echo "Finished: $(date)"
    echo "Total: $total | Success: $successful | Failed: $failed | Skipped: $skipped"
} >> "$summary_log"

log "Complete: $successful/$total succeeded (skipped: $skipped, failed: $failed)"
log "Log: $summary_log"