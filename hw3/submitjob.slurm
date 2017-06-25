#!/bin/bash
#SBATCH --nodes=10
#SBATCH --ntasks=200
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1096
#SBATCH --time=02:00:00
#SBATCH --part=eqa-cs6414
#SBATCH --account=parallelcomputing
#SBATCH --output=output200-1-4

module load mvapich2/gcc
mpiexec ./halo 200 1 10000 10000 4

#sequential run
#./heated_plate 10000 10000 10000
