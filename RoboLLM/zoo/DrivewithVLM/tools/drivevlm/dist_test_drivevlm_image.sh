#!/usr/bin/env bash

# Usage: ./test-distributed.sh <CONFIG_PATH>
export PYTHONPATH=/home/ldc/Projects/RoboLLM/zoo/MMDrive:$PYTHONPATH

CONFIG=$1

# Specify the GPUs to use (4, 5, 6)
export CUDA_VISIBLE_DEVICES=2,3,4,5


# Number of processes per node (number of GPUs)
GPUS_PER_NODE=$2

# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29400}
image_cor_values=(5 3 1)
image_cor_types=("sun_sim" "snow_sim" "fog_sim" "rain_sim" "motion_sim" )
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/drivevlm/"
# Launch the distributed testing using torchrun
for image_cor in "${image_cor_types[@]}"; do
    echo "Testing image corruption type: $image_cor"

    grep -q 'image_cor = "' $CONFIG || { echo "Error: 'image_cor' not found in $CONFIG"; exit 1; }
    sed -i "s/image_cor = \".*\"/image_cor = \"$image_cor\"/" $CONFIG

    for image_cor_value in "${image_cor_values[@]}"; do

        echo "Running with $image_cor = $image_cor_value"

        grep -q "image_cor_value =" $CONFIG || { echo "Error: 'image_cor_value' not found in $CONFIG"; exit 1; }

        sed -i "s/image_cor_value = [0-9]\+/image_cor_value = $image_cor_value/" $CONFIG
        current_time=$(date +"%Y%m%d_%H%M%S")

        torchrun \
            --nproc_per_node=$GPUS_PER_NODE \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $(dirname "$0")/test_ddp_drivevlm_corruption.py \
            --model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/drivevlm/checkpoint-13358' \
            --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/LLM/llava-next-interleave' \
            --result_path="vis/result_${current_time}.json"\
            --pkl_path="vis/pkl/result_${current_time}.pkl"\
            $CONFIG \
            2>&1 | tee ${WORK_DIR}/test.${image_cor}_${image_cor_value}_${current_time}
        done
    done
