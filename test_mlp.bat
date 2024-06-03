@echo off
setlocal


set MODEL_NAME=FaceNet_finetuning_MLP
set PRETRAINED=FaceNet_finetuning_MLP_100_32_deepfake_40
set MODEL_DIR=work_dir\%MODEL_NAME%_06_03_23_11\%PRETRAINED%.pth

set MODEL_NAME=FaceNet_finetuning_Random_Batch
set BATCH_SIZE=32
set EPOCH=30

python test_mlp.py --trained_model %MODEL_DIR% --work_dir test_result/%PRETRAINED%

endlocal