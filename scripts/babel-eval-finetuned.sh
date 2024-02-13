#!/bin/bash
#SBATCH --job-name=pred-gloss-lm
#SBATCH --output ./slurm-out/pred-byt5-finetuned-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=1:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

# ft_glottocode="gitx1241"
# lang_code="git"
# test_split="OOD"

ft_glottocode="dido1241"
lang_code="ddo"
test_split="ID"

# ft_glottocode="uspa1245"
# lang_code="usp"
# test_split="ID"

# exp_name="byt5-translation-all-v2"
exp_name="byt5-baseline-unseg"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/${lang_code}-${exp_name}/"

cd "./src"
python3 pretrain_multilingual_model.py \
    --mode predict \
    --exp_name $exp_name \
    --pretrained_model ${pretrained_model} \
    --ft_glottocode ${ft_glottocode} \
    --test_split ${test_split}

python3 eval.py \
    --pred ../preds/${lang_code}-${exp_name}/test_${test_split}-preds.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_${test_split}
