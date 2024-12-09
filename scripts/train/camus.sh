#!/bin/bash
# Description: Training script for VLSM adapter on Kvasir Polyp dataset

exp=clipseg_dense_adapter_vl
datamodule=camus
exp_name=${exp}_${datamodule}

python src/train.py \
    experiment=$exp \
    datamodule=$datamodule \
    experiment_name=$exp_name \
    datamodule.batch_size=32 \
    model.optimizer.lr=0.001 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    trainer.devices=[0] \
    prompt_type=random \
    seed=0 \
    logger=wandb.yaml \
    output_masks_dir=output_masks/${exp}/${datamodule}/