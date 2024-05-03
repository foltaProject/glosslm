#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output ./slurm-out/eval-%j.out
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --time=5:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --partition=general

source ~/.bashrc
conda init bash
conda activate text2gloss

cd "./src"

python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/shu-crf/test_ID-preds.postprocessed.csv \
    --test_split test_ID

python3 eval.py \
    --pred /home/ltjuatja/glosslm/preds/shu-crf/test_OOD-preds.postprocessed.csv \
    --test_split test_OOD
