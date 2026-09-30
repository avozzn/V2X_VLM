#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29800}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
image_cor_types=("snow_sim")
image_cor_values=(5 3 1)
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/OmniDrive-main/work_dirs/"
RESULTS_BASEDIR="./results_planning_only"
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

for image_cor in "${image_cor_types[@]}"; do
    echo "Testing image corruption type: $image_cor"

    # 先修改 config 里的 image_cor
    grep -q 'image_cor = "' $CONFIG || { echo "Error: 'image_cor' not found in $CONFIG"; exit 1; }
    sed -i "s/image_cor = \".*\"/image_cor = \"$image_cor\"/" $CONFIG

    # 内层循环：遍历不同程度的 image_cor_value
    for image_cor_value in "${image_cor_values[@]}"; do

        echo "Running with $image_cor = $image_cor_value"
        current_time=$(date +"%Y%m%d_%H%M%S")

         # 确保 `config` 里有 `image_cor_value =` 这行
        grep -q "image_cor_value =" $CONFIG || { echo "Error: 'image_cor_value' not found in $CONFIG"; exit 1; }

        # 修改 `image_cor_value` 的值
        sed -i "s/image_cor_value = [0-9]\+/image_cor_value = $image_cor_value/" $CONFIG

        save_path="${RESULTS_BASEDIR}/${image_cor}_${image_cor_value}_${current_time}/"

        grep -q "save_path_value =" $CONFIG || { echo "Error: 'save_path_value' not found in $CONFIG"; exit 1; }

        new_suffix="${image_cor}_${image_cor_value}_${current_time}"

        echo "new_suffix: $new_suffix"

        sed -i "s/new_suffix = \".*\"/new_suffix = \"$new_suffix\"/" $CONFIG

    
        mkdir -p "${save_path}"

        python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --use_env \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            $(dirname "$0")/test.py \
            $CONFIG \
            $CHECKPOINT \
            --launcher pytorch \
            ${@:4} \
            2>&1 | tee ${WORK_DIR}logs/test.${image_cor}_${image_cor_value}_${current_time}

        echo "Starting evaluation..."
        python ./evaluation/eval_planning.py --pred_path="${save_path}" 2>&1 | tee ${WORK_DIR}logs/eval.${image_cor}_${image_cor_value}_${current_time}
        done
    done