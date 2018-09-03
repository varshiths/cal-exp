#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
	--test 2 \
	--resnet_size 28-10 \
	--model_dir savedmodels/den_0.5 \
	--variant den \
	--lamb 0.5 \
	--dummy 0
	# --train_epochs 120 \
	# --epochs_per_eval 1 \
	# --ood_dataset gnoise \
