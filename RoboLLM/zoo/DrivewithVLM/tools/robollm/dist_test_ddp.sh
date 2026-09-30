#!/usr/bin/env bash

# Usage: ./test-distributed.sh <CONFIG_PATH>
export PYTHONPATH=/home/ldc/Projects/RoboLLM/zoo/MMDrive:$PYTHONPATH

CONFIG=$1

# Specify the GPUs to use (4, 5, 6)
export CUDA_VISIBLE_DEVICES=0,1

# Number of processes per node (number of GPUs)
GPUS_PER_NODE=$2

# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29200}
current_time=$(date +"%Y%m%d_%H%M%S")
# Launch the distributed testing using torchrun
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test-ddp_token.py \
    --model_path='/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/MMdrive/checkpoint-8950' \
    --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/LLM/llava-next-interleave' \
    --result_path="vis/mmdrive_senor_sim1_${current_time}.json"\
    --pkl_path="vis/pkl/result_${current_time}.pkl"\
    $CONFIG

