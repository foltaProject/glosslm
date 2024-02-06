#!/bin/bash
#SBATCH --job-name=pred-gloss-lm
#SBATCH --output ./slurm-out/pred-byt5-translation-all-v2-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=6:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

exp_name="byt5-translation-all-v2"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/${exp_name}/"

cd "./src"
for test_split in ID OOD
do
    python3 pretrain_multilingual_model.py \
        --mode predict \
        --exp_name $exp_name \
        --pretrained_model ${pretrained_model} \
        --test_split ${test_split}
    python3 eval.py \
        --pred ../preds/${exp_name}/test_${test_split}-preds.csv
        --test_split test_${test_split}
done