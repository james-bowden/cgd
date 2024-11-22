#!/bin/bash

for data in "control" "ifn" "cocult"
do
	for model in "linear" "linearlr" "mlplr"
	do
		srun -G 1 python run_perturbseq_linear.py --data-path $data --model $model &
		srun -G 1 python run_perturbseq_linear.py --data-path $data --model $model --constraint-mode none &
	done
done
wait # hopefully this lets me still cancel via ctrl-c
