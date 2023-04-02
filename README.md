# IK-NLP Project -- Evaluating Explanations in Natural Language Inference

Contains all code and data used in the project

## Overview

The fine-tuning, prediction and evaluation tasks can all be carried out by running the `nli.py` together with the respective arguments.

The general task is to predict an inference label in the form of `entailment`, `neutral` or `contradiction`, given a premise and a hypothesis.

We split up the general task of NLI into three sub-tasks:

Task | Definition | Prompt
|---|---|---|
| 0 | Predicting the inference label from the premise and hypothesis. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]."
| 1 | Generating an explanation from the premise, hypothesis and inference label. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]. label: [LABEL]"
| 2 | Jointly predicting the inference label and generating an explanation from the premise and hypothesis. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]."

Task | Model (e-SNLI) | Model (custom)
|---|---|---|
| 0 | [rug-nlp-nli/flan-base-nli-label](https://huggingface.co/rug-nlp-nli/flan-base-nli-label) | [rug-nlp-nli/flan-base-nli-label-custom](https://huggingface.co/rug-nlp-nli/flan-base-nli-label-custom)
| 1 | [rug-nlp-nli/flan-base-nli-explanation](https://huggingface.co/rug-nlp-nli/flan-base-nli-explanation) | [rug-nlp-nli/flan-base-nli-explanation-custom](https://huggingface.co/rug-nlp-nli/flan-base-nli-explanation-custom)
| 2 | [rug-nlp-nli/flan-base-nli-label-explanation](https://huggingface.co/rug-nlp-nli/flan-base-nli-label-explanation) | [rug-nlp-nli/flan-base-nli-label-explanation-custom](https://huggingface.co/rug-nlp-nli/flan-base-nli-label-explanation-custom)

For both the prediction and evaluation, the following

Training and inference was tested and confirmed to be working on the following PyTorch backends:

- CUDA (Nvidia GPUs)
- MPS (Apple)
- CPU

## Setup

```bash
# Set up virtual environment
python3 -m virtualenv ik-nlp
source ik-nlp/bin/activate

# Install required python packages
pip install -r requirements.txt
```

## Preprocessing

Optional argument | Definition | Default value
|---|---|---|
| --dataset_name DATASET_NAME | The name of the dataset to be saved after preprocessing. | esnli_reduced
| --distance_cutoff DISTANCE_CUTOFF | The cutoff value for the edit distance based pattern matching criterion | 13

```bash
python3 python nli.py preprocess --dataset_name "preprocessed_dataset" --distance_cutoff 13
```

## Training

Optional argument | Definition | Default value
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
| --subset_size SUBSET_SIZE | The percentage (from 0 to 1) of the dataset that is used during training. | 1
| --custom_dataset CUSTOM_DATASET | Name of the custom dataset. | -

```bash
python3 nli.py train --logging_steps 100 --save_steps 10000 --eval_steps 200  --task 1
```

## Prediction

Note: arguments in boldface are required.
(Optional) argument | Definition | Default value
|---|---|---|
| __--premise PREMISE__ | The premise string. | -
| __--hypothesis HYPOTHESIS__ | The hypothesis string. | -
| --label {entailment,neutral,contradiction} | The corresponding label. | -
| --task TASK | The type of model used for inference. | 2

```bash
python3 nli.py predict --premise "The dog's barking woke up the cat" --hypothesis "the feline was sleeping" --task 2
```

## Evaluation
Optional argument | Definition | Default value
|---|---|---|
| __--model_type MODEL__ | Which model shall be evaluated (standard or custom). | standard
| __--eval_type EVAL_TYPE__ | The type of model evaluation. | both
| --gpu | Run inference on GPU. | True

```bash
python3 nli.py evaluate --model_type "custom" --eval_type "neural"
```
