#!/bin/bash
echo $1

virtual_env_dir=${1:-".venv"}
requirements_file=${2:-"requirements.txt"}

# if requirements_file contains 'dai', set a variable
if [[ "$requirements_file" == *"dai"* ]]; then
    is_delta_ai=true
else
    is_delta_ai=false
fi

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
if [ ! -d "$virtual_env_dir" ]; then
    echo "Creating $virtual_env_dir..."
    uv venv -p 3.10 $virtual_env_dir
else
    echo "$virtual_env_dir already exists"
fi

# Activate the virtual environment
echo "Activating $virtual_env_dir..."
. $virtual_env_dir/bin/activate


uv pip install -r $requirements_file

# install icefall only if is_delta_ai is false
# must be installed after the main requirements to 
# avoid conflicts with espnet
if [[ "$is_delta_ai" == false ]]; then
    if [ ! -d "icefall" ]; then
        echo "Installing icefall..."
        git clone https://github.com/k2-fsa/icefall
    fi
    cd icefall
    rm -rf .git
    uv pip install -r requirements.txt
    export PYTHONPATH=$(pwd):$PYTHONPATH
    cd ..
fi

echo "Setup complete."