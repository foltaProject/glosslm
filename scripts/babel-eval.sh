#!/bin/bash
#SBATCH --job-name=pred-gloss-lm
#SBATCH --output ./slurm-out/pred-glosslm-normalized-all_st_unseg-%j.out
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=6:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

exp_name="glosslm-baseline-normalized-all_st_unseg"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/${exp_name}/"

cd "./src"
for test_split in ID OOD
do
    # python3 pretrain_multilingual_model_normalized.py \
    #     --mode predict \
    #     --exp_name $exp_name \
    #     --pretrained_model ${pretrained_model} \
    #     --exclude_st_seg True \
    #     --test_split ${test_split}
    python3 eval.py \
        --pred ../preds/${exp_name}/test_${test_split}-preds.postprocessed.csv \
        --test_split test_${test_split}
done