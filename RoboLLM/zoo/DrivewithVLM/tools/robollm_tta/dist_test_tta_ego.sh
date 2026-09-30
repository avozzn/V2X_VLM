#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29388}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/tta"
NUM_GPUS=$GPUS
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

export PYTHONPATH
MODEL_ID=llava-interleave-qwen-7b             
TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=8                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=8                                            # the lora alpha (both llm and vision encoder)


DS_STAGE=zero2                                          # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=1                                 # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=1

LR=2e-7                                              # learning rate
MODEL_MAX_LEN=1024                                      # maximum input length of the model
# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes


corr_severity=("mid" "low")

WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/tta"

for corr_severity in "${corr_severity[@]}"; do

    echo "Running with ego = $corr_severity"

    current_time=$(date +"%Y%m%d_%H%M%S")


    # Launch the distributed testing using torchrun
    RUN_ID=${MODEL_ID}
    torchrun $DISTRIBUTED_ARGS \
        $(dirname "$0")/test_ddp_tta.py \
        $CONFIG \
        --model_id $MODEL_ID \
        --model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/MMdrive/checkpoint-8950' \
        --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/LLM/llava-next-interleave' \
        --output_dir ./checkpoints/$RUN_ID \
        --run_name $RUN_ID \
        --report_to wandb \
        --result_path="vis/result_tta_ego_corruption_${corr_ego_type}_${current_time}.json"\
        --pkl_path="vis/pkl/result_tta_ego_corruption_${corr_ego_type}_${current_time}.pkl"\
        --bf16 True \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_train_epochs $NUM_EPOCHS \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate ${LR} \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "constant"\
        --tf32 True \
        --model_max_length $MODEL_MAX_LEN \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --train_vision_encoder $TRAIN_VISION_ENCODER \
        --use_vision_lora $USE_VISION_LORA \
        --train_vision_projector $TRAIN_VISION_PROJECTOR \
        --use_lora $USE_LORA \
        --q_lora $Q_LORA \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA\
        --output_dir ./checkpoints/TTA/$RUN_ID \
        --deepspeed ./tools/ds_configs/${DS_STAGE}.json \
        --corr_ego_type True\
        --corr_severity $corr_severity\
        2>&1 | tee ${WORK_DIR}/test._ego_corruption_${corr_ego_type}_${corr_severity}_${current_time}
    done


