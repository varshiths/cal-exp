#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 main.py \
	--test 0 \
	--resnet_size 28-10 \
	--model_dir savedmodels/temp \
	--variant viby \
	--batch_size 32 \
	--dummy 0
	# --epochs_per_eval 1 \
	# --ood_dataset tinz \
	# --lamb 0.5 \
	# --train_epochs 120 \
