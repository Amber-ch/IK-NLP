#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=gpu
#SBATCH --job-name=NLI-training
#SBATCH --mem=64G

module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load OpenSSL/1.1

cd  /home4/$USER/IK-NLP
source /home4/$USER/IK-NLP/ik-nlp/bin/activate

python nli.py train --gpu --num_train_epochs 3 --logging_steps 100 --save_steps 100000 --eval_steps 400 --batch_size 32 --custom_dataset "esnli_reduced" --task 2

deactivate
