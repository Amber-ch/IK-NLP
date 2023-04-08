# IK-NLP Project -- Evaluating Explanations in Natural Language Inference

This repository provides the source code for training and evaluating natural language inference (NLI) models, as well as running predictions on single inputs.

## Overview

The fine-tuning and preprocessing, prediction and evaluation tasks can all be carried out by running the `nli.py` script together with their respective arguments.

The general task is to predict an inference label in the form of `entailment`, `neutral` or `contradiction`, given a premise and a hypothesis.

We split up the general task of NLI into three sub-tasks, that are each carried out on fine-tuned models, which are based on the [flan-T5-base](https://huggingface.co/google/flan-t5-base) model. The tasks are outlined as follows:

Task | Definition | Input prompt
|---|---|---|
| 0 | Predicting the inference label from the premise and hypothesis. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]."
| 1 | Generating an explanation from the premise, hypothesis and inference label. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]. label: [LABEL]"
| 2 | Jointly predicting the inference label and generating an explanation from the premise and hypothesis. | "premise: [PREMISE]. hypothesis: [HYPOTHESIS]."

In the remaining sections there will be an in-depth explanation on how to properly fine-tune the base model, but to make testing and evaluation easily accessible, we published the two classes of models for the 3 sub-tasks on huggingface: The first class of models was trained on the full e-SNLI dataset, and the second class of models was trained on a subset of the e-SNLI dataset, where a certain percentage of uninformative examples were filtered out using various templates. The details of this are explained in the project report, but for transparency, a Levenshtein cutoff distance of 13 was considered for preprocessing the dataset.

Task | Model (full e-SNLI) | Model (subset e-SNLI)
|---|---|---|
| 0 | [rug-nlp-nli/flan-base-nli-label](https://huggingface.co/rug-nlp-nli/flan-base-nli-label) | [rug-nlp-nli/flan-base-nli-label-custom](https://huggingface.co/rug-nlp-nli/flan-base-nli-label-custom)
| 1 | [rug-nlp-nli/flan-base-nli-explanation](https://huggingface.co/rug-nlp-nli/flan-base-nli-explanation) | [rug-nlp-nli/flan-base-nli-explanation-custom](https://huggingface.co/rug-nlp-nli/flan-base-nli-explanation-custom)
| 2 | [rug-nlp-nli/flan-base-nli-label-explanation](https://huggingface.co/rug-nlp-nli/flan-base-nli-label-explanation) | [rug-nlp-nli/flan-base-nli-label-explanation-custom](https://huggingface.co/rug-nlp-nli/flan-base-nli-label-explanation-custom)

Training and inference was tested and confirmed to be working on the following PyTorch backends:

- CUDA (Nvidia GPU)
- MPS (Apple GPU)
- CPU

## Setup

It is recommended to use Python version 3.8 and higher (but earlier versions may work as well up to a certain point). The best way to set up the workspace is by initializing a virtual environment and installing all the python requirements through pip:

```bash
# Set up virtual environment
python -m virtualenv ik-nlp
source ik-nlp/bin/activate

# Install required python packages
pip install -r requirements.txt
```

## Preprocessing

This is an optional step, and is specifically there to download and reduce the size of the e-SNLI dataset by filtering out uninformative explanations. The degree to which these explanations are matched and then removed is based on the cutoff value for the Levenshtein metric that is used to compare all training example explanations against all templates. This could be considered as the pattern matching 'strictness' (where lower means more strict), and is necessary because most explanations will not be entirely identical to their respective templates.

The program can take any of the following arguments:

Optional argument | Definition | Default value
|---|---|---|
| --dataset_name DATASET_NAME | The name of the dataset to be saved after preprocessing. | esnli_reduced
| --distance_cutoff DISTANCE_CUTOFF | The cutoff value for the edit distance based pattern matching criterion. | 13

The resulting dataset will then be stored in the `data` directory. For the sake of simplicity, we provide the preprocessed dataset with default values as `data/esnli_reduced`. An example of how the preprocessing program could be run with non-default values is as follows:

```bash
python nli.py preprocess --dataset_name "preprocessed_dataset" --distance_cutoff 11
```

## Training

To train a model for one of the three aforementioned sub-tasks, a specific NLP base model should be considered for fine-tuning, and a dataset with training, validation and testing splits should be specified. By default, we consider the [flan-T5-base](https://huggingface.co/google/flan-t5-base) base model from huggingface, but any of its variants or even other models could be considered.

Optional argument | Definition | Default value
|---|---|---|
| --model_dir MODEL_DIR | Name of the model directory. | flan-t5-base-nli-explanation-generation
| --base_model BASE_MODEL | Name of the base model. | google/flan-t5-base
| --gpu | Train on a GPU device. | False
| --batch_size BATCH_SIZE | Training and evaluation batch size. | 32
| --num_train_epochs NUM_TRAIN_EPOCHS | Number of training epochs. | 3
| --eval_steps EVAL_STEPS | Number of evaluation steps. | 100
| --logging_steps LOGGING_STEPS | Number of logging steps. | 20
| --save_steps SAVE_STEPS | Number of steps until saving checkpoint. | 100
| --task {0,1,2} | The type of model being trained. | 2
| --subset_size SUBSET_SIZE | The percentage (from 0 to 1) of the dataset that is used during training. | 1
| --custom_dataset CUSTOM_DATASET | Name of the custom dataset. | -

The resulting model will be stored under the `data/[MODEL_DIR]` directory and is named according to the date and time of training completion. The tensorboard statistics will be stored under the `data/runs` folder.

Other hyperparameters, such as weight decay and the learning rate, are hardcoded at `4e-5` and `0.01` respectively since they are a safe option for the fine-tuning of models for NLI tasks. An example of how the training program could be run with non-default values is as follows:

```bash
python nli.py train --logging_steps 100 --save_steps 10000 --eval_steps 200  --task 1
```

## Prediction

Prediction simply means model inference, and it is used for programatically running an NLI model according to the aforementioned sub-tasks.

__Note__: arguments in __boldface__ are required. Moreover, there is a distinction between the 'custom' option in the `model_type` argument, which refers to the uninformative explanation filtered dataset, and 'custom' in specifying the `custom_model` argument, which refers to the name of the model that is locally stored on the computer.

(Optional) argument | Definition | Default value
|---|---|---|
| __--premise PREMISE__ | The premise string. | -
| __--hypothesis HYPOTHESIS__ | The hypothesis string. | -
| --label {entailment,neutral,contradiction} | The corresponding label. | -
| --task {0,1,2} | The type of model used for inference. | 2
| --model_type {standard, custom} | The type of model used in the prediction. | standard
| --custom_model CUSTOM_MODEL | Name of the locally stored model. | -

It is also important to note that the label argument is not strictly enforced, but should be specified when run for task 1. An example of predicting the label and explanation for a premise and hypothesis pair can be done as follows:

```bash
python nli.py predict --premise "The dog's barking woke up the cat" --hypothesis "the feline was sleeping" --task 2
```

Likewise, predicting the explanation given a premise, hypothesis and label can be done as follows:

```bash
python nli.py predict --premise "The dog's barking woke up the cat" --hypothesis "the feline was sleeping" --label entailment --task 1
```

## Evaluation

The evaluation of a model's explanation over the test split of the dataset can be carried out using two classes of metrics:

1. Neural-based explanation evaluation, using the [BARTScore](https://github.com/neulab/BARTScore) metric.
2. Text-based explanation evaluation, using the [ROUGE-{1,2,L}](https://huggingface.co/spaces/evaluate-metric/rouge) metrics.

Based on the evaluation class, the program generates the following output csv files:

1. `[MODEL_NAME]_neural_scores.csv`, consisting of the test set explanations and their neural scores, sorted in descending order. 
2. `[MODEL_NAME]_rogue_scores.csv`, consisting of the test set explanations and the respective ROUGE-{1,2,L} scores, as well as the max scores across the different example explanations.

Both evaluations also come with a summary text file that reports the average score and standard deviation, as well as the percentage of correctly labeled predictions (for tasks 0 and 2). A final csv file is generated called `allResults.csv`, which is a concatenation of all results, and can be used for analysis.

__Note__: There is a distinction between the 'custom' option in the `model_type` argument, which refers to the uninformative explanation filtered dataset, and 'custom' in specifying the `custom_model` argument, which refers to the name of the model that is locally stored on the computer.

Optional argument | Definition | Default value
|---|---|---|
| --model_type {standard, custom} | The type of model that shall be evaluated. | standard
| --eval_type {neural, text, both} | The type of model evaluation. | both
| --gpu | Run inference on GPU. | True
| --subset_size SUBSET_SIZE | The percentage (from 0 to 1) of the dataset that is used during evaluation. | 1
| --custom_label_model CUSTOM_LABEL_MODEL | Name of the locally stored label model. | -
| --custom_explanation_model CUSTOM_EXPLANATION_MODEL | Name of the locally stored explanation model. | -
| --custom_label_explanation_model CUSTOM_LABEL_EXPLANATION_MODEL | Name of the locally stored label and explanation model. | -

An example of how the evaluation program could be run with non-default values and manually loaded models is as follows:

```bash
python nli.py evaluate --subset_size 0.001 --model_type custom --custom_label_model "label" --custom_explanation_model "exp" --custom_label_explanation_model "label-exp"
```

## Running on HPC (GPU) Cluster

This section may be useful to those that have access to a high performance computing (HPC) GPU cluster that supports SLURM, and want to run the training or evaluation procedure on there. We provide scripts for the different tasks in the `slurm_jobs` directory, and they can be executed on the cluster as follows:

```bash
sbatch slurm_jobs/[TASK_NAME].sh
```

The program output will be piped into the file `slurm-[JOB_ID].out`, and the runtime statistics can be obtained by running:

```bash
squeue --job [JOB_ID]
```

## Jupyter Notebooks

In the `notebooks` directory there is a collection of Jupyter notebooks that are used for exploratory data analysis, template matching analysis and the model evaluations analysis.

__Note__: It is recommended to run the jupyter server from within the notebooks folder as to correctly load the python modules and additional files.
