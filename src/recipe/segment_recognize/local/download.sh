#!/usr/bin/env bash
# Download alignment datasets (HF) and THCHS-30 speech (OpenSLR) for the
# segment_recognize recipe. Run on ghx4-interactive (needs internet) after
# `source setup_uv.sh .venv_dai requirements-dai.txt` (or any shell with
# `.venv_dai/bin` on PATH).
#
# Usage:
#   bash src/recipe/segment_recognize/local/download.sh \
#        [--with-cv-ali] [--force] [--downloads-dir exp/downloads]

set -euo pipefail

WITH_CV_ALI=false
FORCE=false
DOWNLOADS_DIR="exp/downloads"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-cv-ali)   WITH_CV_ALI=true; shift;;
        --force)         FORCE=true; shift;;
        --downloads-dir) DOWNLOADS_DIR="$2"; shift 2;;
        -h|--help)       sed -n '2,9p' "$0"; exit 0;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

# Pin HF hub cache + token to the project's fast filesystem so we don't blow
# up the home-dir quota. Honour an existing HF_HOME if the user set one.
export HF_HOME="${HF_HOME:-$PWD/exp/cache/hf}"
mkdir -p "$HF_HOME"
if [[ ! -f "$HF_HOME/token" && -f "$HOME/.cache/huggingface/token" ]]; then
    cp -f "$HOME/.cache/huggingface/token" "$HF_HOME/token"
    echo "[hf] copied token from \$HOME/.cache/huggingface/token to $HF_HOME"
fi
echo "[hf] HF_HOME=$HF_HOME"

command -v huggingface-cli >/dev/null || {
    echo "huggingface-cli missing; activate .venv_dai first" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl missing" >&2; exit 1; }
mkdir -p "$DOWNLOADS_DIR"

download_hf_dataset() {
    local repo="$1" dir="$2"
    if [[ "$FORCE" == true ]]; then rm -rf "$dir"; fi
    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        echo "[skip] $repo already at $dir"
        return 0
    fi
    echo "[hf] $repo -> $dir"
    huggingface-cli download "$repo" --repo-type dataset --local-dir "$dir"
}

download_openslr_thchs30() {
    local dir="$1"
    local tgz="$dir/data_thchs30.tgz"
    local url="https://www.openslr.org/resources/18/data_thchs30.tgz"
    mkdir -p "$dir"
    if [[ "$FORCE" == true ]]; then rm -f "$tgz" "$dir/.extracted"; fi
    if [[ ! -f "$tgz" ]]; then
        echo "[openslr] $url"
        curl --fail --location --continue-at - --output "$tgz" "$url"
    else
        echo "[skip] $tgz already present"
    fi
    if [[ ! -f "$dir/.extracted" ]]; then
        echo "[extract] $tgz -> $dir"
        tar -xzf "$tgz" -C "$dir"
        touch "$dir/.extracted"
    else
        echo "[skip] already extracted"
    fi
}

# 1. THCHS-30 (alignments + speech from OpenSLR)
download_hf_dataset anyspeech/THCHS-30-alignments \
    "$DOWNLOADS_DIR/thchs30_alignments"
download_openslr_thchs30 "$DOWNLOADS_DIR/thchs30_speech"

# 2. LibriSpeech MFA (alignments only - speech directory is user-provided)
download_hf_dataset anyspeech/librispeech_MFA_alignments \
    "$DOWNLOADS_DIR/librispeech_mfa_alignments"
echo "[note] LibriSpeech speech not downloaded here -"
echo "       point training config at an existing LibriSpeech mirror."

# 3. CommonVoice cv_ali (opt-in; ~4.15 GB)
if [[ "$WITH_CV_ALI" == true ]]; then
    echo "[warn] charsiu/cv_ali is large (~4.15 GB, 6.3M rows)."
    download_hf_dataset charsiu/cv_ali "$DOWNLOADS_DIR/cv_ali_alignments"
else
    echo "[skip] cv_ali alignments (pass --with-cv-ali to include)"
fi
echo "[note] CommonVoice speech not downloaded here - user-provided."

echo "Done. $DOWNLOADS_DIR contents:"
ls -l "$DOWNLOADS_DIR"
