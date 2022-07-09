#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="outputtest"
#SBATCH --exclusive
#SBATCH --nodelist=artemis2
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --account=parallelcomputing

dim=10000
srun mmm $dim $dim $dim $dim 
