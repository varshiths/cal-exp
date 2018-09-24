#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 main.py \
	--test 4 \
	--resnet_size 28-10 \
	--model_dir savedmodels/temp \
	--variant cen \
	--batch_size 32 \
	--epochs_per_eval 1 \
	--ood_dataset tinz \
	--lamb 0.5 \
	--dummy 0
	# --train_epochs 120 \
