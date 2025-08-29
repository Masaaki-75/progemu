#!/bin/bash

WORKSPACE="path/to/your/progemu"
cd "${WORKSPACE}/inference"

export PYTHONPATH=$(pwd)
IMAGE_DIR="directory/to/icg-cxr-v1"
MODEL_PATH="${WORKSPACE}/weights/visiontokenizer"


python infer_vq.py \
    --model_path ${MODEL_PATH} \
    --image_dir ${IMAGE_DIR} \
    --ext '.png' 
