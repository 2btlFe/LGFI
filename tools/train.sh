

# MODEL_NAME=FaceNet
# python inference.py 

#----------------------------------------------------------------------#


# MODEL_NAME=FaceNet_finetuning_Random_Batch_FNP
# BATCH_SIZE=32
# EPOCH=30
# MAIN_LOSS=triplet
# AUX_LOSS=fnp
# WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}

# python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS}
# python inference.py --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}


# #----------------------------------------------------------------------#

MODEL_NAME=FaceNet_finetuning_Random_Batch_L2SP
BATCH_SIZE=32
EPOCH=60
MAIN_LOSS=triplet
AUX_LOSS=l2sp
WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}

python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS}
python inference.py --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}


#----------------------------------------------------------------------#

# MODEL_NAME=FaceNet_finetuning_Random_Batch_RandReg
# BATCH_SIZE=32
# EPOCH=20
# MAIN_LOSS=triplet
# AUX_LOSS=randreg
# WORK_DIR=work_dir/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}

# python train.py --train_dir ./Face_Dataset/Train --val_dir ./Face_Dataset/Validation --batch_size ${BATCH_SIZE} --num_epochs ${EPOCH} --model_name ${MODEL_NAME} --aux_loss ${AUX_LOSS}
# python inference.py --work_dir ${WORK_DIR} --pretrained ${WORK_DIR}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME} 


