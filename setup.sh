#!/bin/bash
# setup.sh — install into .venv from pyproject.toml with uv.
#
# Usage:
#   bash setup.sh          # full install; auto-detects x86_64 vs aarch64
#   bash setup.sh x86      # force the x86 extra (ESPnet + k2)
#   bash setup.sh dai      # force the dai extra (ESPnet, aarch64)
#   bash setup.sh core     # SPAM, MFA and Koel only; no ESPnet
set -e
# Large CUDA wheels exceed uv's 30 s default download timeout on slow networks.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install ffmpeg if missing (via pixi)
if ! command -v ffmpeg >/dev/null 2>&1; then
    command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash
    pixi global install ffmpeg
fi

# Determine extra: arg takes precedence, otherwise detect from arch
if [[ -n "$1" ]]; then
    EXTRA="$1"
elif [[ "$(uname -m)" == "aarch64" ]]; then
    EXTRA="dai"
else
    EXTRA="x86"
fi

echo "Platform: $(uname -m) → installing extra: $EXTRA"

# Create venv if needed and sync deps
if [[ "$EXTRA" == "core" ]]; then
  uv sync
else
  uv sync --extra "$EXTRA"
fi

# ESPnet is a platform-specific fork outside the lockfile; `uv sync` removes it,
# so re-run this script (not bare `uv sync`) after changing pyproject.toml.
if [[ "$EXTRA" == "x86" ]]; then
  uv pip install "espnet @ git+https://github.com/y00njaekim/espnet.git@hotfix"
elif [[ "$EXTRA" == "dai" ]]; then
  uv pip install "espnet @ git+https://github.com/Shikhar-S/espnet.git@flashattn"
fi

source .venv/bin/activate
mkdir -p exp/slurm_logs   # #SBATCH -o targets must exist before the first sbatch

# Delta-AI only: install flash_attn_3 from pre-built egg (aarch64, Hopper GPUs)
# Package name is flash_attn_3 (not flash_attn); uv does not support .egg — use pip.
if [[ "$EXTRA" == "dai" ]]; then
    FLASH_EGG="${FLASH_ATTN3_EGG:-}"
    if [[ -f "$FLASH_EGG" ]]; then
        echo "Installing flash_attn_3 from $FLASH_EGG"
        .venv/bin/pip install "$FLASH_EGG"
    else
        echo "WARNING: FLASH_ATTN3_EGG is unset or missing ($FLASH_EGG); skipping flash_attn_3."
        echo "Build one with: git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention/hopper && python setup.py bdist_egg"
    fi
fi

echo "Setup complete. Environment: .venv (extra=$EXTRA)"
echo "Activate with: source .venv/bin/activate"
