#!/bin/bash
# Description: Training script for VLSM adapter on Kvasir Polyp dataset

exp=clipseg_dense_adapter_vl
datamodule=kvasir_polyp
exp_name=${exp}_${datamodule}

python src/train.py \
    experiment=$exp \
    datamodule=$datamodule \
    experiment_name=$exp_name \
    datamodule.batch_size=32 \
    model.optimizer.lr=1e-3 \
    trainer.accelerator=gpu \
    trainer.precision=16-mixed \
    trainer.devices=[2] \
    prompt_type=random \
    seed=0 \
    logger=wandb.yaml \
    tags='[kvasir_polyp, vlsm_adpter]' \
    output_masks_dir=output_masks/${exp}/${datamodule}/