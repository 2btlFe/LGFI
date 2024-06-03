


MODEL_NAME=FaceNet_finetuning_Random_Batch_L2SP_CE_Frozen_deepfake_total
BATCH_SIZE=32
MAIN_LOSS=triplet   
AUX_LOSS=l2sp       # [fnp, l2sp, randreg]
START_EPOCH=0.8

for EPOCH in 60 70 80 90
do
  WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}
  python train.py --train_dir ./Face_Dataset/Train_deepfake --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS} --start_epoch ${START_EPOCH}
  python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}
done







