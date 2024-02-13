#!/bin/bash
#SBATCH --job-name=git-all-finetune
#SBATCH --output ./slurm-out/git-finetune-byt5-translation-all_st_unseg-v2-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=48GB
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

ft_glottocode="gitx1241"
lang_code="git"

# ft_glottocode="natu1246"
# lang_code="ntu"

# ft_glottocode="nyan1302"
# lang_code="nyb"

model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/"

exp_name="finetune-${ft_glottocode}-patience_15-max_epochs_500-byt5-translation-all_st_unseg-v2"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/byt5-translation-all_st_unseg-v2/"

echo ${lang_code} ${ft_glottocode} ${exp_name} ${pretrained_model}

cd "./src"
python3 git-pretrain_multilingual_model.py \
    --mode finetune \
    --exp_name $exp_name \
    --ft_glottocode ${ft_glottocode} \
    --output_model_path ${model_dir}${exp_name} \
    --max_epochs 500 \
    --pretrained_model ${pretrained_model} \
    --exclude_st_seg True
