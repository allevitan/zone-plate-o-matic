#!/bin/bash
#SBATCH -p day
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16

module load anaconda/2023-06-19
source /opt/psi/TOMCAT/anaconda/2023-06-19/conda/etc/profile.d/conda.sh
conda activate zpom


simulate-sunflower-rzp-focus-t $1 $2 $3 $4 $5 $6 -n $SLURM_ARRAY_TASK_ID --tile_size 4096 --cache-raster --start-time $7 --end-time $8 --time-step $9
