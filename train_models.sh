#!/bin/bash
#SBATCH -J train_gpt_metaphors
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate venv

python3 train_lm_models.py gpt2 --out_path=".experiments/gpt2/"
