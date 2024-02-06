#!/bin/bash
#SBATCH --job-name=glosslm-all-v2
#SBATCH --output ./slurm-out/byt5-translation-all-v2-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=96GB
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=long

source ~/.bashrc
conda init bash
conda activate text2gloss

model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/"
# checkpoint_path="/home/ltjuatja/glosslm/src/training-checkpoints/checkpoint-31474"
exp_name="2_1-byt5-translation-all-v2"
exclude_st_seg="False"

echo $exp_name
echo $exclude_st_seg

cd "./src"
python3 pretrain_multilingual_model.py \
    --mode train --exp_name ${exp_name} \
    --output_model_path ${model_dir}${exp_name} \
    --exclude_st_seg ${exclude_st_seg}
