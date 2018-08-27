#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
	--test 1 \
	--model_dir savedmodels/temp \
	--ood_dataset noise \
	--resnet_size 28-10 \
	--dummy 0
	# --hinged True \

	# --resnet_size 50 \
	# --model_dir savedmodels/wrn28-10-unhinged-cifar10-none \
	# --model_dir savedmodels/cifar10_resnet50_original \
	# --data_dir /mnt/blossom/data/ujjwaljain/cifar10_data/ \

# --test flag:
# check main.py