#!/bin/bash

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
