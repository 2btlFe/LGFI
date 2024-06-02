#python inference.py --pretrained /workspace/ssd0/byeongcheol/FaceID/model/Facenet512_transfer_learning_20_32_deepfake.pth
# python inference.py


MODEL_NAME=FaceNet_finetuning_Random_Batch
BATCH_SIZE=32
EPOCH=30

python inference.py --pretrained /workspace/ssd0/byeongcheol/FaceID/model/${MODEL_NAME}/${MODEL_NAME}_${EPOCH}_${BATCH_SIZE}_deepfake.pth --model_name ${MODEL_NAME}

