#!/bin/bash
#SBATCH --job-name=finetune-norm
#SBATCH --output ./slurm-out/finetune-norm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
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
# checkpoint_path="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-normalized/training_checkpoints/dido1241/st_unseg-normalized/checkpoint-28"

# ft_glottocode="uspa1245"
# lang_code="usp"
# checkpoint_path="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-normalized/training_checkpoints/uspa1245/st_unseg-normalized/checkpoint-152"

# ft_glottocode="lezg1247"
# lang_code="lez"
# checkpoint_path="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-normalized/training_checkpoints/lezg1247/st_unseg-normalized/checkpoint-44"

# ft_glottocode="natu1246"
# lang_code="ntu"
# checkpoint_path="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-normalized/training_checkpoints/natu1246/st_unseg-normalized/checkpoint-49"

# ft_glottocode="nyan1302"
# lang_code="nyb"
# checkpoint_path="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-normalized/training_checkpoints/nyan1302/st_unseg-normalized/checkpoint-82"

ft_glottocode="gitx1241"
lang_code="git"
max_epochs=500
early_stopping_patience=15


save_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-normalized/"

exp_name="st_unseg-normalized"
pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/glosslm-normalized-all_st_unseg/"

echo ${lang_code} ${ft_glottocode} ${exp_name} ${pretrained_model}

cd "./src"
python3 pretrain_multilingual_model_normalized.py \
    --mode finetune \
    --exp_name $exp_name \
    --ft_glottocode ${ft_glottocode} \
    --output_model_path ${save_dir}${ft_glottocode} \
    --pretrained_model ${pretrained_model} \
    --exclude_st_seg True \
    --use_translation True \
    --max_epochs ${max_epochs} \
    --early_stopping_patience ${early_stopping_patience} \
    --checkpoint_save_dir "${save_dir}training_checkpoints/${ft_glottocode}" \