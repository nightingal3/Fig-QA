#!/bin/bash
#SBATCH -J gpt_metaphors
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --exclude=tir-0-19,tir-1-7,tir-1-11,tir-1-18,tir-1-23,tir-1-28
#SBATCH --output=gpt-sm-interject.txt

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate venv

# libgcc issue?
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/

python3 gpt_score.py gpt-neo-sm --middle_phrase="That is to say, " --out_path="./gpt-neo-sm-interject"
