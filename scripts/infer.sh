#!/bin/bash

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

WORKSPACE="path/to/your/progemu"
cd "${WORKSPACE}/inference"

export PYTHONPATH=$(pwd)
DATA_PATH="path/to/your/data"
EXP_NAME="progemu-init"
OUTPUT_DIR="logs/${EXP_NAME}"
MODEL_PATH="path/to/your/model"


python run_inference.py \
    --model_path ${MODEL_PATH} \
    --meta_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --use_pc_proc \
    --do_sample 
