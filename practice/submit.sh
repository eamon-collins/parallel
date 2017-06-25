#!/bin/bash
# This is a template for a simple SLURM sbatch job script file
#
# First, SBATCH options - these could be passed via the command line, but this
# is more convenient
#
#SBATCH --job-name="avgarr" #Name of the job which appears in squeue
#
#SBATCH --error="my_job.err"                    # Where to write std err
#SBATCH --output="output"                # Where to write stdout
#SBATCH --nodelist=hermes[1-4]

srun pthread