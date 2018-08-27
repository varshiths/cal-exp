#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
	--test 0 \
	--model_dir savedmodels/temp \
	--ood_dataset noise \
	--resnet_size 28-10 \
	--hinged True \
	--dummy 0
	# --resnet_size 50 \

	# --model_dir savedmodels/cifar10_resnet50_original \
	# --data_dir /mnt/blossom/data/ujjwaljain/cifar10_data/ \

# --test flag:
# check main.py