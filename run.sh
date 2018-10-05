#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python3 main.py \
    --test 4 \
    --resnet_size 28-10 \
    --batch_size 32 \
    --ood_dataset tinz \
    --model_dir savedmodels/viby_160_0.005 \
    --variant viby \
    --dim_z 160 \
    --dummy 0
    # --train_epochs 90 \
    # --lamb 1.0 \
    # --epochs_per_eval 1 \
