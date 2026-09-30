
#!/usr/bin/env bash

# Usage: ./test-distributed.sh <CONFIG_PATH>
export PYTHONPATH=/home/ldc/Projects/RoboLLM/zoo/MMDrive:$PYTHONPATH

CONFIG=$1

# Specify the GPUs to use (4, 5, 6)
Checkpoint=$2
# Number of processes per node (number of GPUs)
GPUS_PER_NODE=$3




DS_STAGE=zero3
# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29398}
current_time=$(date +"%Y%m%d_%H%M%S")


USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=8                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=8                                            # the lora alpha (both llm and vision encoder)
LORA_DROPOUT=0.05
LORA_BIAS="none"

MODEL_ID=llava-interleave-qwen-7b 
# Launch the distributed testing using torchrun
deepspeed  \
    --include localhost:1,2\
    $(dirname "$0")/test_deepspeed.py \
    --model_path=$Checkpoint \
    --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/LLM/llava-next-interleave' \
    --result_path="vis/result_${current_time}.json"\
    --pkl_path="vis/pkl/result_${current_time}.pkl"\
    --use_lora=$USE_LORA \
    --q_lora=$Q_LORA \
    --lora_r=$LORA_R \
    --lora_alpha=$LORA_ALPHA\
    --lora_bias=$LORA_BIAS\
    --lora_dropout=$LORA_DROPOUT\
    --model_id $MODEL_ID \
    --deepspeed=./tools/ds_configs/${DS_STAGE}.json \
    --config=$CONFIG
