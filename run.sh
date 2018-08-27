#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
	--test 0 \
	--resnet_size 28-10 \
	--model_dir savedmodels/wrn28-10-unhinged-cifar10-none \
	--dummy 0
	# --hinged True \
	# --ood_dataset noise \

# --test flag:
# check main.py