#!/bin/bash
#SBATCH --job-name=git-unseg-byt5-baseline
#SBATCH --output ./slurm-out/git-unseg-byt5-baseline-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32GB
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

# ft_glottocode="arap1274"
# lang_code="arp"

# ft_glottocode="dido1241"
# lang_code="ddo"

# ft_glottocode="uspa1245"
# lang_code="usp"

ft_glottocode="gitx1241"
lang_code="git"

model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/"

cd "./src"
# python3 pretrain_multilingual_model.py \
#     --mode finetune \
#     --exp_name byt5-baseline-unseg-${lang_code} \
#     --ft_glottocode ${ft_glottocode} \
#     --output_model_path ${model_dir}{lang_code}-byt5-baseline-unseg \
#     --max_epochs 20 \
#     --pretrained_model "google/byt5-base" \
#     --exclude_st_seg True \
#     --checkpoint_save_dir ${model_dir}training-checkpoints

python3 git-pretrain_multilingual_model.py \
    --mode finetune \
    --exp_name byt5-baseline-unseg-${lang_code} \
    --ft_glottocode ${ft_glottocode} \
    --output_model_path ${model_dir}{lang_code}-byt5-baseline-unseg \
    --max_epochs 200 \
    --pretrained_model "google/byt5-base" \
    --exclude_st_seg True \
    --checkpoint_save_dir ${model_dir}training-checkpoints
