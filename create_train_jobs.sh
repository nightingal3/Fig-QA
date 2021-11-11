#!/bin/bash

set -e

model=$1
# Create jobs file
job_dir=$PWD/job_scripts
mkdir -p ${job_dir}
for seed in `seq 0 9`
do
    for num_epochs in `seq 3 8`
    do
        for lr in "1e-5" "3e-5" "5e-5"
        do
            job_subdir=${job_dir}/${model}
            job_file=${job_dir}/${model}/epochs_${num_epochs}_lr_${lr}_seed_${seed}.job
            mkdir -p ${job_subdir}
            echo "#!/bin/bash
#SBATCH -J train_${model}_metaphors_${num_epochs}_${lr}
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate venv

python3 train_lm_models.py ${model} --num_epochs=${num_epochs} --learning_rate=${lr} --output_path="./experiments/${model}/epochs_${num_epochs}_lr_${lr}/seed_${seed}"
" >> ${job_file}
        done
    done
done
