#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=gpu
#SBATCH --job-name=NLI-training-task-0
#SBATCH --mem=64G

module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load OpenSSL/1.1

cd  /home4/$USER/IK-NLP
source /home4/$USER/IK-NLP/ik-nlp/bin/activate

python nli.py evaluate --eval_type both

deactivate
