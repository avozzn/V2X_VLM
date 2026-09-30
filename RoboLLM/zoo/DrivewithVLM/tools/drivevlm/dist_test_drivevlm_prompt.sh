#!/usr/bin/env bash

# Usage: ./test-distributed.sh <CONFIG_PATH>
export PYTHONPATH=/home/ldc/Projects/RoboLLM/zoo/MMDrive:$PYTHONPATH

CONFIG=$1

# Specify the GPUs to use (4, 5, 6)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Number of processes per node (number of GPUs)
GPUS_PER_NODE=$2

# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29330}
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/drivevlm/" # Set your working directory here
attack_type=( "injection" "delete" "overwrite_rules" "conversation" "charcter")
for attack_type in "${attack_type[@]}"; do

    echo "Running with $attack_type"

        # 确保 `config` 里有 `image_cor_value =` 这行
    grep -q "attack_type =" $CONFIG || { echo "Error: 'attack_type' not found in $CONFIG"; exit 1; }

    sed -i "s/attack_type = \".*\"/attack_type = \"$attack_type\"/" $CONFIG

    current_time=$(date +"%Y%m%d_%H%M%S")

    # Launch the distributed testing using torchrun
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $(dirname "$0")/test_ddp_drivevlm_prompt.py \
        --model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/drivevlm/checkpoint-13358' \
        --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/LLM/llava-next-interleave' \
        --result_path="vis/result_vlm_${attack_type}_${current_time}.json"\
        --pkl_path="vis/pkl/result_vlm_${attack_type}_${current_time}.pkl"\
        $CONFIG \
        2>&1 | tee ${WORK_DIR}/test.${attack_type}_${current_time}
    done
