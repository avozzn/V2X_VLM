#!/usr/bin/env bash

# Usage: ./test-distributed.sh <CONFIG_PATH>
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 

export PYTHONPATH
CONFIG=$1

# Specify the GPUs to use (4, 5, 6)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Number of processes per node (number of GPUs)
GPUS=$2
NUM_GPUS=$GPUS
# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29390}
current_time=$(date +"%Y%m%d_%H%M%S")
NNODES=${NNODES:-1}
DISTRIBUTED_ARGS="
    --nnodes ${NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${PORT}"

# Launch the distributed testing using torchrun
python -m torch.distributed.run $DISTRIBUTED_ARGS \
    $(dirname "$0")/test-ddp_token.py \
    --model_path='/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/checkpoints/V2X_dt_re_vm_9step/checkpoint-1472' \
    --origin_model_path='/home/zzn/V2X_VLM/RoboLLM/zoo/DrivewithVLM/checkpoints/LLM/llava-next-interleave' \
    --result_path="vis/result_dt_re_${current_time}.json"\
    --pkl_path="vis/pkl/result_dt_re_${current_time}.pkl"\
    $CONFIG
