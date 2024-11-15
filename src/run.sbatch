#!/bin/bash

#SBATCH -J checker 
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=20

srun --gpu-bind=single:1 python -m main
exit 0