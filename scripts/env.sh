#!/bin/bash
# Activate the project venv and expose the NVIDIA runtime libraries that the
# torch +cu126 wheel dlopens. Sourced by every scripts/run_*.sh after cd to
# the repo root; run `make install` once beforehand.
source .venv/bin/activate
# scripts/*.py import the src package from the repo root.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
NVIDIA_LIB_DIRS=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia" \
  -maxdepth 3 -name lib -type d 2>/dev/null | tr '\n' ':')
[[ -n "$NVIDIA_LIB_DIRS" ]] && export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:-}"
# Run scripts point HF_HOME at exp/cache/hf, so the token saved by
# `huggingface-cli login` is not found there; pass it through explicitly.
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]] && export HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"
