#!/bin/bash
#SBATCH --job-name=pred-gloss-lm
#SBATCH --output ./slurm-out/pred-glosslm-norm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=5:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

exp_name="glosslm-finetune-normalized-all_st_unseg"

# declare -A ID_langs=( [arp]="arap1274" ["ddo"]="dido1241" ["usp"]="uspa1245")
# declare -A ID_langs=( [arp]="arap1274" )
# declare -A OOD_langs=( [git]="gitx1241" [lez]="lezg1247" [ntu]="natu1246" [nyb]="nyan1302" )
declare -A OOD_langs=( [git]="gitx1241" )


for lang in "${!OOD_langs[@]}" 
do
    echo $lang
    ft_glottocode="${OOD_langs[$lang]}"
    pretrained_model="/data/tir/projects/tir6/general/ltjuatja/glosslm/finetune-normalized/${ft_glottocode}"
    test_split="OOD"
    cd "./src"
    python3 pretrain_multilingual_model_normalized.py \
        --mode predict \
        --exp_name $exp_name \
        --pretrained_model ${pretrained_model} \
        --ft_glottocode ${ft_glottocode} \
        --test_split ${test_split}
    python eval.py \
        --pred ../preds/${exp_name}/${lang}/${lang}-preds.postprocessed.csv \
        --test_split test_${test_split}
done

