#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output="output64"
#SBATCH --nodelist=hermes1
#SBATCH --mem_bind=local
#SBATCH --exclusive

#SBATCH --time=02:00:00
#SBATCH --account=parallelcomputing

srun pthread 64 10000 10000 
