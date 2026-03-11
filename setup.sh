#!/bin/bash
# setup.sh — preferred install script using pyproject.toml + uv sync
# Retains setup_uv.sh / requirements*.txt for reference; this is the new path.
#
# Usage:
#   bash setup.sh          # auto-detects x86_64 vs aarch64
#   bash setup.sh x86      # force x86 extra (Delta / Babel)
#   bash setup.sh dai      # force dai extra (Delta-AI)
set -e

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
uv sync --extra "$EXTRA"
source .venv/bin/activate

# x86 only: clone and install icefall (not pip-installable)
if [[ "$EXTRA" == "x86" ]]; then
    if [ ! -d icefall ]; then
        echo "Cloning icefall..."
        git clone https://github.com/k2-fsa/icefall
        rm -rf icefall/.git
    fi
    uv pip install -r icefall/requirements.txt
    export PYTHONPATH="$(pwd)/icefall:$PYTHONPATH"
    echo "PYTHONPATH updated. Add this to your shell profile to persist:"
    echo "  export PYTHONPATH=\"$(pwd)/icefall:\$PYTHONPATH\""
fi

# Delta-AI only: install flash_attn_3 from pre-built egg (aarch64, Hopper GPUs)
# Package name is flash_attn_3 (not flash_attn); uv does not support .egg — use pip.
if [[ "$EXTRA" == "dai" ]]; then
    FLASH_EGG="/work/nvme/bbjs/sbharadwaj/powsm/flash-attention/hopper/dist/flash_attn_3-3.0.0b1-py3.10-linux-aarch64.egg"
    if [[ -f "$FLASH_EGG" ]]; then
        echo "Installing flash_attn_3 from $FLASH_EGG"
        .venv/bin/pip install "$FLASH_EGG"
    else
        echo "WARNING: flash_attn_3 egg not found at $FLASH_EGG"
        echo "Rebuild with: cd /work/nvme/bbjs/sbharadwaj/powsm/flash-attention/hopper && python setup.py bdist_egg"
    fi
fi

echo "Setup complete. Environment: .venv (extra=$EXTRA)"
echo "Activate with: source .venv/bin/activate"
