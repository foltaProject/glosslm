#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:3
#SBATCH --ntasks=3     # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=train_glosslm.%j.out      # Output file name
#SBATCH --error=train_glosslm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/glosslm/src"

exp_name="unimorph-unseg-distributed"
exclude_st_seg="True"
use_unimorph="True"

torchrun --nproc_per_node=4 pretrain_multilingual_model.py \
    --mode train --exp_name ${exp_name} \
    --output_model_path /projects/migi8081/glosslm/models/${exp_name} \
    --exclude_st_seg ${exclude_st_seg} \
    --use_unimorph ${use_unimorph}