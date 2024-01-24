#!/bin/bash
#SBATCH --job-name=glosslm
#SBATCH --output ./slurm-out/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=babel-shared

source ~/.bashrc
conda init bash
conda activate text2gloss

model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/"
exp_name="byt5-translation-all"

echo $exp_name

cd "./src"
<<<<<<< Updated upstream
python3 pretrain_multilingual_model.py --mode train --output_model_path ${model_dir}${exp_name}
=======
python3 pretrain_multilingual_model.py --mode predict --model_path ${model_dir}${exp_name} --test_split "id"
python3 pretrain_multilingual_model.py --mode predict --model_path ${model_dir}${exp_name} --test_split "ood"
>>>>>>> Stashed changes
