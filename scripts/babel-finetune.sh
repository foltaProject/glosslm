#!/bin/bash
#SBATCH --job-name=nyb-all-finetune
#SBATCH --output ./slurm-out/nyb-finetune-byt5-translation-all-v2-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --mem=96GB
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=long

source ~/.bashrc
conda init bash
conda activate text2gloss

# ft_glottocode="natu1246"
# lang_code="ntu"

ft_glottocode="nyan1302"
lang_code="nyb"

model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/"

exp_name="finetune-${ft_glottocode}-byt5-translation-all-v2"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/byt5-translation-all-v2/"

echo ${lang_code} ${ft_glottocode} ${exp_name} ${pretrained_model}

cd "./src"
python3 pretrain_multilingual_model.py \
    --mode finetune \
    --exp_name $exp_name \
    --ft_glottocode ${ft_glottocode} \
    --output_model_path ${model_dir}finetuned-${lang_code}-byt5-translation-all-v2 \
    --max_epochs 100 \
    --pretrained_model ${pretrained_model}