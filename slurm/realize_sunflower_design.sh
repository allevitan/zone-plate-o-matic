#!/bin/bash
#SBATCH -p hour #gpu-week
#SBATCH -A gpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
# #SBATCH --gpus-per-task 1
#SBATCH --time=00:30:00

module load anaconda/2023-06-19
source /opt/psi/TOMCAT/anaconda/2023-06-19/conda/etc/profile.d/conda.sh
conda activate zpom

# For GPU
#design-sunflower-rzp $1 -n $SLURM_ARRAY_TASK_ID --device 'cuda' --tile_size 4098 -y
# For CPU
realize_sunflower_design $1 -n $SLURM_ARRAY_TASK_ID --tile_size 2048 -y
