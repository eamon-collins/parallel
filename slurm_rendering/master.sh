#!/bin/bash

NUM_PROCS=$1
CURR_PROC=1
while [ $CURR_PROC -lt $(($NUM_PROCS+1)) ];  
do
	sbatch render2.slurm $NUM_PROCS $CURR_PROC
	((CURR_PROC++))
done
