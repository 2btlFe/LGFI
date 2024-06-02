
MODEL_NAME=FaceNet_finetuning_Random_Batch_RandReg
BATCH_SIZE=32
MAIN_LOSS=triplet
AUX_LOSS=randreg

for EPOCH in 5 10 20 30 40 50 60 70
do
    WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}
    python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS}
    python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME} 
done



