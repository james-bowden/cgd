# Causal graph discovery final project: Stat256 (Berkeley), Fall 2024

This codebase begins with a fork of ["Large-Scale Differentiable Causal Discovery of Factor Graphs"](https://github.com/Genentech/dcdfg/tree/main).
This is a group project by James Bowden, Carlos Guirado, and Hanyang Li.

This repository contains the experiments on both the Perturb-CITE-seq and Sachs protein datasets.

# *Everything below is from the original repo, to be edited*.

## Requirements

Python 3.9+ is required. To install the requirements:
```setup
pip install -r requirements.txt
```
You may also have to add the environment library to path, with 
```
export LD_LIBRARY_PATH=/path_to_env/env_name/lib/:$LD_LIBRARY_PATH
```
Additionally, updated torch may be required in order to run on your version of GPU.
wandb is required for now (a PR to make remove this requirement is welcome). Follow the steps [here](https://docs.wandb.ai/quickstart).


## Running DCD-FG

### SEMs simulations (full usage in files)
1. 'python make_lowrank_dataset.py'
2. 'python run_gaussian.py'
### Biological dataset
1. 'perturb-cite-seq/0-data-download.ipynb'
1. 'perturb-cite-seq/1-assignments-vs-variability.ipynb'
2. 'python run_perturbseq_linear.py'

## Acknowledgments
- This repository was originally forked from [DCDI](https://github.com/slachapelle/dcdi). Please refer to the license file for more information.
- Most files in this codebase have been rewritten for:
1. vectorization and scaling to large graphs
2. incorporating the semantic of factor graphs
3. refactoring and implementation in pytorch lightning
4. implementation of DCD-FG, NOTEARS, NOTEARS-LR and NOBEARS
- We are grateful to the authors of the baseline methods for releasing their code.
