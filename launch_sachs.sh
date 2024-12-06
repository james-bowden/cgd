#!/bin/bash

# srun -G 1 python run_sachsprotein.py --model linear &
# srun -G 1 python run_sachsprotein.py --model linear --constraint-mode none &

for nm in 1 2 3 4 10
do
	for model in "linearlr" "mlplr"
	do
		# srun -G 1 python run_sachsprotein.py --model $model --num-modules $nm &
		# srun -G 1 python run_sachsprotein.py --model $model --num-modules $nm --constraint-mode none &
		python run_sachsprotein.py --model $model --num-modules $nm
		python run_sachsprotein.py --model $model --num-modules $nm --constraint-mode none
	done
done
wait # hopefully this lets me still cancel via ctrl-c
