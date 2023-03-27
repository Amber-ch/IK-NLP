# IK-NLP Project

Contains all code and data used in the project

## Overview

The fine-tuning, prediction and evaluation tasks can all be carried out by running the `nli.py` together with the respective arguments arguments.

The general task is to predict an inference label in the form of `entailment`, `neutral` or `contradiction`, given a premise and a hypothesis.

We split up the general task of NLI into three sub-tasks:

Task | Definition | Prompt
|---|---|---|
| 0 | Predicting the inference label from the premise and hypothesis. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]."
| 1 | Generating an explanation from the premise, hypothesis and inference label. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]. label: [LABEL]"
| 2 | Jointly predicting the inference label and generating an explanation from the premise and hypothesis. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]."

Task | Model
|---|---|---|
| 0 | [rug-nlp-nli/flan-base-nli-label](https://huggingface.co/rug-nlp-nli/flan-base-nli-label)
| 1 | [rug-nlp-nli/flan-base-nli-explanation](https://huggingface.co/rug-nlp-nli/flan-base-nli-explanation)
| 2 | [rug-nlp-nli/flan-base-nli-label-explanation](https://huggingface.co/rug-nlp-nli/flan-base-nli-label-explanation)

For both the prediction and evaluation, the following

Training and inference was tested on the following PyTorch backends:

- CUDA
- MPS
- CPU

## Setup

```bash
# Set up virtual environment
python3 -m virtualenv ik-nlp
source ik-nlp/bin/activate

# Install required python packages
pip install -r requirements.txt
```

## Training

(Optional) argument | Definition | Default value
|---|---|---|
| --model_dir MODEL_DIR | Name of the model directory. | flan-t5-base-nli-explanation-generation
| --base_model BASE_MODEL | Name of the base-model. | google/flan-t5-base
| --gpu | Train on a GPU device. | False
| --batch_size BATCH_SIZE | Training and evaluation batch size. | 32
| --num_train_epochs NUM_TRAIN_EPOCHS | Number of training epochs. | 3
| --eval_steps EVAL_STEPS | Number of evaluation steps. | 100
| --logging_steps LOGGING_STEPS | Number of logging steps. | 20
| --save_steps SAVE_STEPS | Number of steps until saving checkpoint. | 100
| --task TASK | The type of model being trained. | 2

```bash
python3 nli.py train --logging_steps 100 --save_steps 10000 --eval_steps 200  --task 2
```

## To cite

TODO
