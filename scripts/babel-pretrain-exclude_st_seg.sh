#!/bin/bash
#SBATCH --job-name=glosslm-all_st_unseg-v2
#SBATCH --output ./slurm-out/byt5-translation-all_st_unseg-v2-%j.out
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
checkpoint_path="/home/ltjuatja/glosslm/src/training-checkpoints/2_1-byt5-translation-all_st_unseg-v2/checkpoint-28613"
exp_name="2_5-byt5-translation-all_st_unseg-v2-cont"
exclude_st_seg="True"

echo $exp_name
echo $exclude_st_seg

cd "./src"
python3 pretrain_multilingual_model.py \
    --mode train \
    --exp_name ${exp_name} \
    --output_model_path ${model_dir}${exp_name} \
    --exclude_st_seg ${exclude_st_seg} \
    --checkpoint_path ${checkpoint_path}
