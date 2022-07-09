#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1096
#SBATCH --time=00:35:00 
#SBATCH --part=eqa-cs6414
#SBATCH --account=parallelcomputing
#SBATCH --output=script-output

# the number of frames to be used to generate the video
NUM_PROCS=$1
PROC_ID=$2

# load the renderer engine that will generate frames for the video 
module load blender


# -b option is for command line access, i.e., without an output console
# -s option is the starting frame
# -e option is the ending frame
# -a option indicates that all frames should be rendered
iter=$PROC_ID
frames=""
while [ $iter -lt 251 ];  #251
do
frames="$frames-f $iter "
iter=$(($iter+$NUM_PROCS))
done

#create a directory in tmp for each job and store a copy of the .blend file in it
srun mkdir /tmp/ec3bd
srun mkdir /tmp/ec3bd/"$SLURM_JOB_ID"
srun cp Star-collapse-ntsc.blend /tmp/ec3bd/"$SLURM_JOB_ID"/Star-collapse-ntsc.blend

#only access this processors copy of the blend file so they aren't all trying to open the same file
blender -b /tmp/ec3bd/"$SLURM_JOB_ID"/Star-collapse-ntsc.blend -o /scratch/ec3bd/HW2/frames/star-collapse_ -t 1 $frames

if [ $(ls frames/star-collapse_* | wc -l) -eq 250 ]; then
	# need to give the generated frames some extension; otherwise the video encoder will not work
	ls frames/star-collapse_* | xargs -I % mv % %.jpg

	# load the video encoder engine
	module load ffmpeg

	# start number should be 1 as by default the encoder starts looking from file ending with 0
	# frame rate and start number options are set before the input files are specified so that the
	# configuration is applied for all files going to the output
	ffmpeg -framerate 25 -start_number 1 -i frames/star-collapse_%04d.jpg -vcodec mpeg4 output.avi
fi
#cleaning up after myself
srun rm -rf /tmp/ec3bd
