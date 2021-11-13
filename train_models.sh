#!/bin/bash
#SBATCH -J train_gpt_metaphors
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --exclude=tir-0-19
#SBATCH --gres=gpu:v100:1

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate venv

# libgcc issue?
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/

python3 train_lm_models.py gpt-neo-lg --seed=42 --cuda
