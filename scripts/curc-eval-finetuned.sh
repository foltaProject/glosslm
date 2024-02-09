#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=preds_glosslm.%j.out      # Output file name
#SBATCH --error=preds_glosslm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/glosslm/src"

for glottocode in arap1274 dido1241 uspa1245 nyan1302 natu1246 lezg1247
do
  python3 pretrain_multilingual_model.py --mode predict --pretrained_model /rc_scratch/migi8081/models/byt5-baseline-${glottocode} --test_split id --ft_glottocode $glottocode
done