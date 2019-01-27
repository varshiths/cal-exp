#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
    --test 0 \
    --resnet_size 28-10 \
    --model_dir savedmodels/base \
    --variant odin \
    --batch_size 128 \
    --train_epochs 200 \
    --dummy 0
    # --train_epochs 110 \
    # --ood_dataset tinz \
    # --pen_prob 0.10 \
    # --cutoff_weight 1.0 \
