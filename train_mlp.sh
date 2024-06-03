#!/bin/bash

# 환경 변수 설정
MODEL_NAME="FaceNet_finetuning_MLP"
BATCH_SIZE=32
MAIN_LOSS="triplet"
# AUX_LOSS="l2sp"       # [fnp, l2sp, randreg]
START_EPOCH=0.8
EPOCH=100
SAVE_PERIOD=20

WORK_DIR="work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}"
mkdir -p $WORK_DIR  # WORK_DIR 생성

# train_mlp.py 실행
python3 train_mlp.py --train_dir ./Face_Dataset/Train_deepfake --val_dir ./Face_Dataset/Validation --batch_size $BATCH_SIZE --num_epochs $EPOCH --model_name $MODEL_NAME --start_epoch $START_EPOCH --save_period $SAVE_PERIOD
# python3 train_mlp.py --train_dir ./Face_Dataset/Train_deepfake --val_dir ./Face_Dataset/Validation --batch_size $BATCH_SIZE --num_epochs $EPOCH --model_name $MODEL_NAME --aux_loss $AUX_LOSS --start_epoch $START_EPOCH --save_period $SAVE_PERIOD
# python3 inference.py --work_dir $WORK_DIR --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name $MODEL_NAME
