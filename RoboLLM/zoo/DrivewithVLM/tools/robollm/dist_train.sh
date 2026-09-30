#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${GPU_IDS:-4,5}
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29511}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=$GPUS
DISTRIBUTED_ARGS="
    --nnodes ${NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${PORT}"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

export PYTHONPATH
# arguments that are very likely to be changed
# according to your own case
MODEL_ID=llava-interleave-qwen-7b                                # model id; pick on by running `python supported_models.py`               
TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=8                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=8                                            # the lora alpha (both llm and vision encoder)

RUN_ID=${RUN_ID:-V2X_dt_9step_4096} # keeps runs and seeds separate

DS_STAGE=zero2                                          # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=1                                 # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=${NUM_EPOCHS:-6}                             # number of training epochs
SEED=${SEED:-42}

LR=2e-4                                               # learning rate
MODEL_MAX_LEN=4096                                      # multimodal sequence length, including the answer

python -m torch.distributed.run $DISTRIBUTED_ARGS \
    $(dirname "$0")/train.py \
    $CONFIG \
    --model_id $MODEL_ID \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to wandb \
    --run_name $RUN_ID \
    --deepspeed ./tools/ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 6 \
    --seed ${SEED} \
    --data_seed ${SEED} \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
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
    --lora_alpha $LORA_ALPHA
