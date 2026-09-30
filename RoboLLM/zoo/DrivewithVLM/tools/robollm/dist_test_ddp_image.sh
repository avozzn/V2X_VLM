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
MASTER_PORT=${MASTER_PORT:-29220}
image_cor_values=(1)
image_cor_types=("dark_sim")
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/mmdrive/"
# Launch the distributed testing using torchrun
# 外层循环：遍历 image_cor 类型
for image_cor in "${image_cor_types[@]}"; do
    echo "Testing image corruption type: $image_cor"

    # 先修改 config 里的 image_cor
    grep -q 'image_cor = "' $CONFIG || { echo "Error: 'image_cor' not found in $CONFIG"; exit 1; }
    sed -i "s/image_cor = \".*\"/image_cor = \"$image_cor\"/" $CONFIG

    # 内层循环：遍历不同程度的 image_cor_value
    for image_cor_value in "${image_cor_values[@]}"; do

        echo "Running with $image_cor = $image_cor_value"

         # 确保 `config` 里有 `image_cor_value =` 这行
        grep -q "image_cor_value =" $CONFIG || { echo "Error: 'image_cor_value' not found in $CONFIG"; exit 1; }

        # 修改 `image_cor_value` 的值
        sed -i "s/image_cor_value = [0-9]\+/image_cor_value = $image_cor_value/" $CONFIG
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
            --result_path="vis/result_${current_time}.json"\
            --pkl_path="vis/pkl/result_${current_time}.pkl"\
            $CONFIG \
            2>&1 | tee ${WORK_DIR}image_remain.${image_cor}_${image_cor_value}_${current_time}
        done
    done

