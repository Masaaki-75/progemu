#!/bin/bash

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")


WORKSPACE="path/to/your/progemu"
cd ${WORKSPACE}

export PYTHONPATH=$(pwd)
DATA_PATH="path/to/your/data"
EXP_NAME="progemu-init"
OUTPUT_DIR="logs/${EXP_NAME}"

MODEL_PATH="weights/gen"
TOKENIZER_PATH="weights/gen"

USE_TEMPLATE=False
NUM_TRAIN_EPOCHS=20
BS_PER_GPU=1
GRAD_ACCUM_STEPS=16
SAVE_STEPS=200
LOGGING_STEPS=100
MAX_POS_EMB=9216
LR=1e-5

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    emu3/train/train_ti2ti.py \
    --model_name_or_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --deepspeed scripts/zero3.json \
    --log_level info \
    --data_path ${DATA_PATH} \
    --use_template ${USE_TEMPLATE} \
    --use_registered True \
    --null_prompt_prob 0.01 \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_text False \
    --image_area 262144 \
    --max_position_embeddings ${MAX_POS_EMB} \
    --output_dir ${OUTPUT_DIR} \
    --bf16 True \
    --tf32 True \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${BS_PER_GPU} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LR} \
    --min_learning_rate 1e-6 \
    --weight_decay 0.1 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --warmup_steps 30 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps ${LOGGING_STEPS} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${EXP_NAME}



