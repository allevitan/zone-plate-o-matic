#!/bin/bash
#SBATCH -p day
#SBATCH -A gpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1

module load anaconda/2023-06-19
source /opt/psi/TOMCAT/anaconda/2023-06-19/conda/etc/profile.d/conda.sh
conda activate zpom

collate-sunflower-design $1 
