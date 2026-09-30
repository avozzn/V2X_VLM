#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=offline
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29345}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/ALL"
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

LR=5e-5                                              # learning rate
MODEL_MAX_LEN=1024                                      # maximum input length of the model
# Optional: Set MASTER_ADDR and MASTER_PORT if running on multiple nodes

#"sun_sim" "fog_sim" "rain_sim" "snow_sim" "dark_sim"
image_cor_values=(5)
corr_prompt_severity=("overwrite_rules")
image_cor_types=("snow_sim")
WORK_DIR="/home/ldc/Projects/RoboLLM/zoo/MMDrive/work_dirs/ALL"
# Launch the distributed testing using torchrun
# 外层循环：遍历 image_cor 类型
for image_cor in "${image_cor_types[@]}"; do
    echo "Testing image corruption type: $image_cor"


    # 内层循环：遍历不同程度的 image_cor_value
    for image_cor_value in "${image_cor_values[@]}"; do

        echo "Running with $image_cor = $image_cor_value"

        current_time=$(date +"%Y%m%d_%H%M%S")

  
        # Launch the distributed testing using torchrun
        RUN_ID=${MODEL_ID}
        torchrun $DISTRIBUTED_ARGS \
            $(dirname "$0")/test_all.py \
            $CONFIG \
            --model_id $MODEL_ID \
            --model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/MMdrive/checkpoint-8950' \
            --origin_model_path='/home/ldc/Projects/RoboLLM/zoo/MMDrive/checkpoints/LLM/llava-next-interleave' \
            --output_dir ./checkpoints/$RUN_ID \
            --run_name $RUN_ID \
            --report_to wandb \
            --result_path="vis/result_all_${image_cor}_${image_cor_value}_${current_time}.json"\
            --pkl_path="vis/pkl/result_all_${image_cor}_${image_cor_value}_${current_time}.pkl"\
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
            --output_dir ./checkpoints/ALL/$RUN_ID \
            --deepspeed ./tools/ds_configs/${DS_STAGE}.json \
            --corr_image_type $image_cor\
            --corr_severity $image_cor_value\
            --corr_prompt_type True\
            --corr_prompt_severity $corr_prompt_severity\
            2>&1 | tee ${WORK_DIR}/test.${image_cor}_${image_cor_value}_${corr_prompt_type}_${corr_prompt_severity}_${current_time}
        done
    done

