#!/usr/bin/env bash

# Usage: ./test-distributed.sh <CONFIG_PATH>
export PYTHONPATH=/home/ldc/Projects/RoboLLM/zoo/MMDrive:$PYTHONPATH



# Specify the GPUs to use (4, 5, 6)
export CUDA_VISIBLE_DEVICES=2

# Number of processes per node (number of GPUs)
GPUS_PER_NODE=$1

# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29398}
current_time=$(date +"%Y%m%d_%H%M%S")

# Launch the distributed testing using torchrun
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test_action.py \
    --model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/llava-interleave-qwen-7b_lora-True_qlora-False/checkpoint-26000' \
    --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/LLM/llava-next-interleave' \
    --result_path="vis/result_${current_time}.json"\
    --pkl_path="vis/pkl/result_${current_time}.pkl"\
    --config1='/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/configs/test/origin.py'\
    --config2='/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/configs/test/no_camera.py'\
    --config3='/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/configs/test/no_ego.py'\
    --config4='/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/configs/test/no_image.py'\
    --config5='/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/configs/test/no_text.py'\
    --config6='/home/ldc/Projects/RoboLLM/zoo/MMDrive/projects/configs/test/no_sensor.py'\
