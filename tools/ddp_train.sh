#!/bin/sh

CONFIG=$1
GPUS=$2 
if [ -z $GPUS ]; then
    GPUS=-1
fi

python tools/train.py fit \
    --model.config_path $CONFIG \
    --data.config_path $CONFIG \
    --trainer.accelerator "gpu" \
    --trainer.devices "$GPUS" \
    --trainer.strategy "DDPStrategy" \
    --trainer.strategy.find_unused_parameters "False" \
    --trainer.logger "TensorBoardLogger" \
    --trainer.logger.save_dir "tb_logs" \
    --trainer.logger.name "svtr-ocr" \
    --trainer.max_epochs "-1" \
    --seed_everything 42 \
    --optimizer "AdamW" \
    --optimizer.lr "0.001" \
    --optimizer.weight_decay "0.01" \
    --lr_scheduler "CosineAnnealingLR" \
    --lr_scheduler.T_max "10" \
    --trainer.callbacks+ "LearningRateMonitor" \
    --trainer.callbacks.logging_interval "step" \
    --trainer.callbacks+ "ModelCheckpoint" \
    --trainer.callbacks.monitor "val_loss" \
    --trainer.callbacks.save_top_k "3" \
    --trainer.callbacks.save_last "True" \
    --trainer.callbacks.save_weights_only "True" \
    --trainer.callbacks.mode "min" \
    --trainer.callbacks.filename "{epoch}-{val_loss:.2f}-{val_acc:.2f}" \
    --trainer.callbacks+ "EarlyStopping" \
    --trainer.callbacks.monitor "val_loss" \
    --trainer.callbacks.patience "20" \
    --trainer.callbacks.mode "min" \
