#!/bin/bash
#SBATCH --job-name=glosslm-norm-all
#SBATCH --output ./slurm-out/glosslm-norm-all-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=96GB
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=long

source ~/.bashrc
conda init bash
conda activate text2gloss

model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/"
checkpoint_parent_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/checkpoints/"
exp_name="glosslm-normalized-all"
exclude_st_seg="False"

echo $exp_name
echo $exclude_st_seg

cd "./src"
python3 pretrain_multilingual_model_normalized.py \
    --mode train --exp_name ${exp_name} \
    --output_model_path ${model_dir}${exp_name} \
    --exclude_st_seg ${exclude_st_seg} \
    --checkpoint_save_dir ${checkpoint_parent_dir}${exp_name}