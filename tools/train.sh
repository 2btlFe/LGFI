

# MODEL_NAME=FaceNet
# python inference.py 

#----------------------------------------------------------------------#


# MODEL_NAME=FaceNet_finetuning_Random_Batch_FNP_CE
# BATCH_SIZE=32
# MAIN_LOSS=triplet
# AUX_LOSS=fnp
# WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}

# for EPOCH in  20 30 40 50 60 70 80 90
# do
#   WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}
#   python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS}
#   python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}
# done
# #----------------------------------------------------------------------#

# MODEL_NAME=FaceNet_finetuning_Random_Batch_FNP_CE_Fusion
# BATCH_SIZE=32
# MAIN_LOSS=triplet
# AUX_LOSS=fnp
# WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}
# START_EPOCH=0.8

# for EPOCH in  20 30 40 50 60 70 80 90
# do
#   WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}
#   python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS} --start_epoch ${START_EPOCH}
#   python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}
# done


MODEL_NAME=FaceNet_finetuning_Random_Batch_L2SP_aug
BATCH_SIZE=32
MAIN_LOSS=triplet
AUX_LOSS=l2sp
START_EPOCH=0.8

for EPOCH in  70 80 90
do
  WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}
  python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS} --start_epoch ${START_EPOCH}
  python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}
done

#----------------------------------------------------------------------#

MODEL_NAME=FaceNet_finetuning_Random_Batch_L2SP_aug
BATCH_SIZE=32
MAIN_LOSS=triplet
AUX_LOSS=l2sp
START_EPOCH=0.9

for EPOCH in 70 80 90
do
  WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}
  python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS} --start_epoch ${START_EPOCH}
  python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}
done






# MODEL_NAME=FaceNet_finetuning_Random_Batch_RandReg
# BATCH_SIZE=32
# EPOCH=20
# MAIN_LOSS=triplet
# AUX_LOSS=randreg
# WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}

# python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS}
# python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME} 


