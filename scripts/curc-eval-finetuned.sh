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
python3 pretrain_multilingual_model.py --mode predict --pretrained_model ../models/finetuned-arp --test_split id --ft_glottocode arap1274
python3 pretrain_multilingual_model.py --mode predict --pretrained_model ../models/finetuned-ddo --test_split id --ft_glottocode dido1241
python3 pretrain_multilingual_model.py --mode predict --pretrained_model ../models/finetuned-usp --test_split id --ft_glottocode uspa1245
python3 pretrain_multilingual_model.py --mode predict --pretrained_model ../models/finetuned-nyb --test_split ood --ft_glottocode nyan1302
python3 pretrain_multilingual_model.py --mode predict --pretrained_model ../models/finetuned-ntu --test_split ood --ft_glottocode natu1246
python3 pretrain_multilingual_model.py --mode predict --pretrained_model ../models/finetuned-lez --test_split ood --ft_glottocode lezg1247