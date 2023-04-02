#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --job-name=NLI-training
#SBATCH --mem=64G
 
module purge
module load Python/3.8.6-GCCcore-10.2.0
module load PyTorch/1.10.0-fosscuda-2020b
module load OpenSSL/1.1

python3 nli.py evaluate --eval_type both
