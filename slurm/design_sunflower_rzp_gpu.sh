#!/bin/bash
#SBATCH -p gpu-week
#SBATCH -A gpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --gpus-per-task 1

module load anaconda/2023-06-19
source /opt/psi/TOMCAT/anaconda/2023-06-19/conda/etc/profile.d/conda.sh
conda activate zpom

# For GPU
design-sunflower-rzp $1 -n $SLURM_ARRAY_TASK_ID --device 'cuda' --tile_size 4096 -y

