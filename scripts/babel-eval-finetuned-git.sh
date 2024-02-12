#!/bin/bash
#SBATCH --job-name=pred-gloss-lm
#SBATCH --output ./slurm-out/pred-byt5-translation-all_st_unseg-finetuned-%j.out
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time=2:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

ft_glottocode="gitx1241"
lang_code="git"

exp_name="byt5-translation-all-v2-patience_15-max_epochs_500"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-gitx1241-patience_15-max_epochs_500-byt5-translation-all-v2"
test_split="test_OOD"

cd "./src"
# python3 pretrain_multilingual_model.py \
#     --mode predict \
#     --exp_name $exp_name \
#     --pretrained_model ${pretrained_model} \
#     --ft_glottocode ${ft_glottocode} \
#     --test_split OOD

python3 append_input_to_preds.py \
    --pred_dir ../preds/${lang_code}-${exp_name} \
    --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds.csv \
    --test_split ${test_split} \
    --ft_glottocode ${ft_glottocode}
python3 eval.py \
    --pred ../preds/${lang_code}-${exp_name}/${test_split}-preds-postprocessed.csv \
    --ft_glottocode ${ft_glottocode} \
    --test_split test_OOD
