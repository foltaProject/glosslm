#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu
#SBATCH --ntasks=4          # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=finetune_glosslm.%j.out      # Output file name
#SBATCH --error=finetune_glosslm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

# purge all existing modules
module purge
# Load the python module
module load anaconda
# Run Python Script
conda activate AutoIGT
cd "/projects/migi8081/glosslm/src"
python3 pretrain_multilingual_model.py --mode finetune --ft_glottocode arap1274 --output_model_path ../models/finetuned-arp --pretrained_model lecslab/byt5-translation-all
python3 pretrain_multilingual_model.py --mode finetune --ft_glottocode dido1241 --output_model_path ../models/finetuned-ddo --pretrained_model lecslab/byt5-translation-all
python3 pretrain_multilingual_model.py --mode finetune --ft_glottocode uspa1245 --output_model_path ../models/finetuned-usp --pretrained_model lecslab/byt5-translation-all
python3 pretrain_multilingual_model.py --mode finetune --ft_glottocode nyan1302 --output_model_path ../models/finetuned-nyb --pretrained_model lecslab/byt5-translation-all
python3 pretrain_multilingual_model.py --mode finetune --ft_glottocode natu1246 --output_model_path ../models/finetuned-ntu --pretrained_model lecslab/byt5-translation-all
python3 pretrain_multilingual_model.py --mode finetune --ft_glottocode lezg1247 --output_model_path ../models/finetuned-lez --pretrained_model lecslab/byt5-translation-all
