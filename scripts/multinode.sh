#!/bin/bash

# IMPORTANT: without this wrapper bash script lightning 
# does not spawn processes on multiple nodes correctly
# run with command:
# srun \
#   --account=bbjs-dtai-gh \
#   --partition ghx4-interactive \
#   --nodes=2 \
#   --ntasks-per-node=4 \
#   --cpus-per-task=8 \
#   --gpus-per-node=4 \
#   --mem=100G \
#   --time=00:30:00 \
#   --pty \
#   multinode.sh

export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=hsn
module load nccl

# Navigate to the project directory
cd /work/nvme/bbjs/sbharadwaj/powsm/xeuspr

# Activate the environment
source setup_uv.sh .venv_dai requirements-dai.txt

echo "Running multinode training script."

python -u src/main.py \
      experiment=train/ipapack_xeuspr \
      data.batch_size=16 \
      data.num_workers=12 \
      trainer.strategy=ddp \
      trainer.num_nodes=2 \
      trainer.devices=auto