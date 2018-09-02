#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
	--test 0 \
	--resnet_size 28-10 \
	--model_dir savedmodels/temp \
	--epochs_per_eval 1 \
	--dummy 0
	# --ood_dataset gnoise \
	# --hinged True \
	# --model_dir savedmodels/wrn28-10-unhinged-cifar10-none \
	# --model_dir savedmodels/wrn28-10-hinged-cifar10-tin \
	# --model_dir savedmodels/wrn28-10-hinged-cifar10-gnoise \

# --test flag:
# check main.py