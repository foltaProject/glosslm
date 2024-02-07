#!/bin/bash
#SBATCH --job-name=pred-gloss-lm
#SBATCH --output ./slurm-out/pred-byt5-translation-all-finetuned-%j.out
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --time=2:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

# ft_glottocode="gitx1241"
# lang_code="git"

ft_glottocode="lezg1247"
lang_code="lez"

exp_name="byt5-translation-all-v2"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetuned-${lang_code}-${exp_name}/"

cd "./src"
python3 pretrain_multilingual_model.py \
    --mode predict \
    --exp_name $exp_name \
    --pretrained_model ${pretrained_model} \
    --ft_glottocode ${ft_glottocode} \
    --test_split OOD
python3 eval.py \
    --pred ../preds/${lang_code}-${exp_name}/test_OOD-preds.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_OOD
