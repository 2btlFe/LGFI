@echo off
setlocal

set MODEL_NAME=FaceNet_finetuning_MLP
set BATCH_SIZE=32
set MAIN_LOSS=triplet
rem AUX_LOSS=l2sp       # [fnp, l2sp, randreg]
set START_EPOCH=0.8
set EPOCH=100
set SAVE_PERIOD=20

set WORK_DIR=work_dir\%MODEL_NAME%_%EPOCH%_%BATCH_SIZE%
python train_mlp.py --train_dir .\Face_Dataset\Train_deepfake --val_dir .\Face_Dataset\Validation --batch_size %BATCH_SIZE% --num_epochs %EPOCH% --model_name %MODEL_NAME% --start_epoch %START_EPOCH% --save_period %SAVE_PERIOD%
rem python train_mlp.py --train_dir .\Face_Dataset\Train_deepfake --val_dir .\Face_Dataset\Validation --batch_size %BATCH_SIZE% --num_epochs %EPOCH% --model_name %MODEL_NAME% --aux_loss %AUX_LOSS% --start_epoch %START_EPOCH% --save_period %SAVE_PERIOD%
rem python inference.py --work_dir %WORK_DIR% --pretrained %WORK_DIR%\%MODEL_NAME%_%EPOCH%_%BATCH_SIZE%_deepfake.pth --model_name %MODEL_NAME%

endlocal