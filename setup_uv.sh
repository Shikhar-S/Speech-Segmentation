#!/bin/bash

# Check pixi
if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi not found. Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
else
    echo "pixi is already installed"
fi

# Check uv
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | bash
else
    echo "uv is already installed"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    pixi global install ffmpeg
fi

# If .venv doesn't exist, create it
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    uv venv -p 3.10
else
    echo ".venv already exists"
fi

# Activate the virtual environment
echo "Activating .venv..."
. .venv/bin/activate


uv pip install -r requirements.txt
