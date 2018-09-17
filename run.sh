#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
	--test 0 \
	--resnet_size 28-10 \
	--model_dir savedmodels/cen_1.0 \
	--variant cen \
	--lamb 1.0 \
	--dummy 0
	# --train_epochs 120 \
	# --epochs_per_eval 1 \
	# --ood_dataset gnoise \
