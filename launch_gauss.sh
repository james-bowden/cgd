#!/bin/bash

for data in "data_p100_m10_n50000_nonlinear"
do
	for model in "linear" "linearlr" "mlplr"
	do
		srun -G 1 python run_gaussian.py --data-dir "gaussian_data/$data" --model $model &
		srun -G 1 python run_gaussian.py --data-dir "gaussian_data/$data" --model $model --constraint-mode none &
	done
done
wait # hopefully this lets me still cancel via ctrl-c
