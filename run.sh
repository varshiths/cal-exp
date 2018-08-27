#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 main.py \
	--test 1 \
	--ood_dataset noise \
	--resnet_size 28-10 \
	--model_dir savedmodels/temp \
	--dummy 0

	# --model_dir savedmodels/cifar10_resnet50_original \
	# --data_dir /mnt/blossom/data/ujjwaljain/cifar10_data/ \

# --test flag:
# flag = 0 -> training set with ood
# flag = 1 -> test with ood
# flag = 2 -> only ood
