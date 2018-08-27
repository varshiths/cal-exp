#!/bin/bash
python3 get_insights.py \
	--N 20 \
	--file_main outputs/$1_mmce.out \
	--file_base outputs/$1_baseline.out \
	--file_out $1.png \
	--temperature 1.0
	# --file_main outputs/$1_mmce.out \
	# --file_base outputs/$1_baseline.out \