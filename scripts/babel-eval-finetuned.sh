#!/bin/bash
#SBATCH --job-name=pred-gloss-lm
#SBATCH --output ./slurm-out/pred-byt5-finetuned-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=5:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

exp_name="all-no_trans"
ft_glottocode="arap1274"
lang_code="arp"
test_split="ID"
cd "./src"
# pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-no_trans/finetune-${ft_glottocode}-${exp_name}/"
# echo $ft_glottocode
# cd "./src"
# python3 pretrain_multilingual_model.py \
#     --mode predict \
#     --exp_name $exp_name \
#     --pretrained_model ${pretrained_model} \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split ${test_split}

exp_name="all-no_trans"
ft_glottocode="arap1274"
lang_code="arp"
test_split="ID"

echo $ft_glottocode
python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/glosslm-all-no_trans/${lang_code}-${exp_name}/test_${test_split}-preds.postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_${test_split} \


exp_name="all-no_trans"
ft_glottocode="dido1241"
lang_code="ddo"
test_split="ID"

echo $ft_glottocode
python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/glosslm-all-no_trans/${lang_code}-${exp_name}/test_${test_split}-preds.postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_${test_split} \

exp_name="all-no_trans"
ft_glottocode="uspa1245"
lang_code="usp"
test_split="ID"

echo $ft_glottocode
python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/glosslm-all-no_trans/${lang_code}-${exp_name}/test_${test_split}-preds.postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_${test_split} \


exp_name="all-no_trans"
ft_glottocode="gitx1241"
lang_code="git"
test_split="OOD"

echo $ft_glottocode
python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/glosslm-all-no_trans/${lang_code}-${exp_name}/test_${test_split}-preds.postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_${test_split} \

exp_name="all-no_trans"
ft_glottocode="lezg1247"
lang_code="lez"
test_split="OOD"

echo $ft_glottocode
python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/glosslm-all-no_trans/${lang_code}-${exp_name}/test_${test_split}-preds.postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_${test_split} \


exp_name="all-no_trans"
ft_glottocode="natu1246"
lang_code="ntu"
test_split="OOD"

echo $ft_glottocode
python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/glosslm-all-no_trans/${lang_code}-${exp_name}/test_${test_split}-preds.postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_${test_split} \
