#!/bin/bash
#SBATCH --job-name=arp-all-no_trans
#SBATCH --output ./slurm-out/arp-all-no_trans-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=long

source ~/.bashrc
conda init bash
conda activate text2gloss

ft_glottocode="arap1274"
lang_code="arp"
checkpoint_path="/home/ltjuatja/glosslm/src/training_checkpoints/finetune-arap1274-all-no_trans/all-no_trans/checkpoint-6726"

# ft_glottocode="dido1241"
# lang_code="ddo"

# ft_glottocode="uspa1245"
# lang_code="usp"

# ft_glottocode="lezg1247"
# lang_code="lez"

# ft_glottocode="natu1246"
# lang_code="ntu"

# ft_glottocode="gitx1241"
# lang_code="git"
# max_epochs=500
# early_stopping_patience=15


model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-no_trans/"

exp_name="all-no_trans"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/byt5-translation-all-v2/"
save_dir="finetune-${ft_glottocode}-all-no_trans"

echo ${lang_code} ${ft_glottocode} ${exp_name} ${pretrained_model}

cd "./src"
python3 pretrain_multilingual_model.py \
    --mode finetune \
    --exp_name $exp_name \
    --ft_glottocode ${ft_glottocode} \
    --output_model_path ${model_dir}${save_dir} \
    --pretrained_model ${pretrained_model} \
    --exclude_st_seg False \
    --use_translation False \
    --checkpoint_path $checkpoint_path \
    --checkpoint_save_dir "training_checkpoints/${save_dir}" \
