#!/usr/bin/env bash

# Usage: ./test-distributed.sh <CONFIG_PATH>
export PYTHONPATH=/home/ldc/Projects/RoboLLM/zoo/MMDrive:$PYTHONPATH

CONFIG=$1

# Specify the GPUs to use (4, 5, 6)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Number of processes per node (number of GPUs)
GPUS_PER_NODE=$2

# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/mmdrive"
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29301}
# attack_type=("character" "delete" "role_change" "injection" "overwrite_rulescode" "overwrite_rules")
attack_type=("delete")
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
        $(dirname "$0")/test-ddp_token.py \
        --model_path='/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/MMdrive/checkpoint-8950' \
        --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/DrivewithVLM/checkpoints/LLM/llava-next-interleave' \
        --result_path="vis/result_noLidar_${attack_type}_${current_time}.json"\
        --pkl_path="vis/pkl/result_noLidar_${attack_type}_${current_time}.pkl"\
        $CONFIG \
        2>&1 | tee ${WORK_DIR}/test.mmdrive_${attack_type}_${current_time}
    done

