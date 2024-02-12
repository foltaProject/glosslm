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

cd "./src"

exp_name="byt5-translation-all-v2"

# ft_glottocode="arap1274"
# lang_code="arp"
# test_split="test_ID"
# python3 eval.py \
#     --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds-postprocessed.csv \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split ${test_split} \


# ft_glottocode="dido1241"
# lang_code="ddo"
# test_split="test_ID"
# python3 eval.py \
#     --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds-postprocessed.csv \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split ${test_split}


# ft_glottocode="uspa1245"
# lang_code="usp"
# test_split="test_ID"
# python3 eval.py \
#     --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds-postprocessed.csv \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split ${test_split}


ft_glottocode="lezg1247"
lang_code="lez"
test_split="test_OOD"
python3 append_input_to_preds.py \
    --pred_dir ../preds/${lang_code}-${exp_name} \
    --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds.csv \
    --test_split ${test_split} \
    --ft_glottocode ${ft_glottocode}
python3 eval.py \
    --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds-postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split ${test_split}


# ft_glottocode="natu1246"
# lang_code="ntu"
# test_split="test_OOD"
# python3 eval.py \
#     --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds-postprocessed.csv \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split ${test_split}


# ft_glottocode="nyan1302"
# lang_code="nyb"
# test_split="test_OOD"
# python3 eval.py \
#     --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds-postprocessed.csv \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split ${test_split}


# python3 pretrain_multilingual_model.py \
#     --mode predict \
#     --exp_name $exp_name \
#     --pretrained_model ${pretrained_model} \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split OOD
