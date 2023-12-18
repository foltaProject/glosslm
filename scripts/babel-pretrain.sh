#!/bin/bash

#SBATCH --job-name=glosslm
#SBATCH --output ./slurm-out/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=lindiat@andrew.cmu.edu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=babel-shared-long

source ~/.bashrc
conda init bash
conda activate text2gloss

model_dir="/data/tir/projects/tir6/general/ltjuatja/glosslm/"
exp_name="byt5_translation_all"

echo $exp_name

cd "./src"
python3 pretrain_multilingual_model.py --mode train --model_path ${model_dir}${exp_name}
