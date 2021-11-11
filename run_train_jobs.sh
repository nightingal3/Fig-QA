#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mail-type=END
#SBATCH --mail-user=mengyan3@andrew.cmu.edu
#SBATCH --time=1-00:00:00
#SBATCH --array=0-179%5

files=(./job_scripts/gpt2/*.job)
job_file=${files[$SLURM_ARRAY_TASK_ID]}
sbatch ${job_file}