#!/bin/bash

srun -G 1 python run_sachsprotein.py --model linear &
srun -G 1 python run_sachsprotein.py --model linear --constraint-mode none &

# for nm in 1 2 3 4 5 10
for nm in 5
do
	for model in "linearlr" "mlplr"
	do
		srun -G 1 python run_sachsprotein.py --model $model &
		srun -G 1 python run_sachsprotein.py --model $model --constraint-mode none &
	done
done
wait # hopefully this lets me still cancel via ctrl-c
