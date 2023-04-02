"""Implements the model fine-tuning.

Part of the code adapated from: https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
"""

import datasets
import numpy as np
import nltk

from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from src.utils import *


# Settings
DATASET_DIR = 'data'


class TrainModel:
    """Implementation for model fine-tuning using a huggingface NLI dataset."""

    labels_dict = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    def __init__(self, param_args, model_dir):
        """Initialize model trainer.

        Args:
            param_args (args): The arguments from argparse.
            model_dir (str): The path for storing the model.
        """

        self.base_model = param_args.base_model
        self.subset_size = param_args.subset_size
        self.device = select_device(param_args.gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._setup_dataset('esnli', param_args.custom_dataset)
        self._preprocess_data(task=param_args.task)

        # Load punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Trainer arguments
        self.args = Seq2SeqTrainingArguments(
            model_dir,
            evaluation_strategy="steps",
            eval_steps=param_args.eval_steps,
            logging_strategy="steps",
            logging_steps=param_args.logging_steps,
            save_strategy="steps",
            save_steps=param_args.save_steps,
            learning_rate=4e-5,
            per_device_train_batch_size=param_args.batch_size,
            per_device_eval_batch_size=param_args.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=param_args.num_train_epochs,
            predict_with_generate=True,
            fp16=False,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            report_to="tensorboard",
            use_mps_device=(str(self.device)=='mps')
        )

        # Trainer utils and metrics
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)

        self.trainer = Seq2SeqTrainer(
            model_init=self._model_init,
            args=self.args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )

    def _setup_dataset(self, name='esnli', custom=None):
        """Set up the dataset

        Args:
            name (str, optional): Name of the Hugging Face model. Defaults to 'esnli'.
            custom (str, optional): Name of the locally cached model. Defaults to None.
        """

        seed = None

        if custom:
            dataset = datasets.load_from_disk(f'{DATASET_DIR}/{custom}')
        else:
            dataset = datasets.load_dataset(name)

        # Take subset of training set
        train_size = int(dataset['train'].shape[0] * self.subset_size)
        val_size = int(dataset['validation'].shape[0] * self.subset_size)
        test_size = int(dataset['test'].shape[0] * self.subset_size)

        self.dataset = datasets.DatasetDict()
        self.dataset['train'] = dataset['train'].shuffle(
            seed).select(range(train_size))
        self.dataset['validation'] = dataset['validation'].shuffle(
            seed).select(range(val_size))
        self.dataset['test'] = dataset['test'].shuffle(
            seed).select(range(test_size))

    def _tokenize(self, example):
        """Tokenize the input and target strings"""

        tokenized = self.tokenizer(example['input'])

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(example['target'])

        tokenized['labels'] = targets['input_ids']

        return tokenized

    def _preprocess_data(self, task=2):
        """Preprocess the dataset

        Format the dataset with inputs that follow the prompt format, and targets according to the task number.
        Finally tokenize the the inputs and targets.

        Args:
            task (int, optional): The task that corresponds to the input prompt. Defaults to 2.
        """

        # Only label as target
        if task == 0:
            updated_dataset = self.dataset.map(lambda example: {"input": "premise: " + example["premise"] + " hypothesis: " + example["hypothesis"], "target": self.labels_dict[example["label"]]}, remove_columns=[
                                               "premise", "hypothesis", "label", "explanation_1", "explanation_2", "explanation_3"])
        # Label in input and explanation1 as target
        elif task == 1:
            updated_dataset = self.dataset.map(lambda example: {"input": "premise: " + example["premise"] + " hypothesis: " + example["hypothesis"] + " label: " + self.labels_dict[
                                               example["label"]], "target": example['explanation_1']}, remove_columns=["premise", "hypothesis", "label", "explanation_1", "explanation_2", "explanation_3"])
        # Label and explanation1 both as targets
        else:
            updated_dataset = self.dataset.map(lambda example: {"input": "premise: " + example["premise"] + " hypothesis: " + example["hypothesis"], "target": self.labels_dict[
                                               example["label"]] + ": " + example['explanation_1']}, remove_columns=["premise", "hypothesis", "label", "explanation_1", "explanation_2", "explanation_3"])

        # Tokenize
        encoded_dataset = updated_dataset.map(
            lambda example: self._tokenize(example), batched=True)
        encoded_dataset.with_format("torch")

        self.dataset = encoded_dataset

    def _compute_metrics(self, eval_pred):
        """Compute training metrics

        Args:
            eval_pred (tuple): Contains predictions and the target label.

        Returns:
            dict: Metrics and their values
        """

        metric = load_metric("rouge")

        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ['\n'.join(nltk.sent_tokenize(pred.strip()))
                         for pred in decoded_preds]
        decoded_labels = ['\n'.join(nltk.sent_tokenize(label.strip()))
                          for label in decoded_labels]

        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id)
                           for pred in predictions]
        result['gen_len'] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def _model_init(self):
        """Function that returns an untrained model to be trained."""

        return AutoModelForSeq2SeqLM.from_pretrained(self.base_model).to(self.device)

    def train(self):
        """Initiates training."""

        self.trainer.train()

    def save(self, model_name):
        """Saves the model."""

        path = f'{DATASET_DIR}/{model_name}/{get_timestamp()}'
        # Save model separately from the training checkpoints
        self.trainer.save_model(path)

    def print_config(self, args):
        """Print the model configuration."""

        print('Dataset')
        print('===========================')
        print(self.dataset)
        print('===========================')

        print('Model configuration')
        print('===========================')
        print(f'eval_steps={args.eval_steps}')
        print(f'logging_steps={args.logging_steps}')
        print(f'save_steps={args.save_steps}')
        print(f'batch_size={args.batch_size}')
        print(f'num_train_epochs={args.num_train_epochs}')
        print('===========================')


def run(args):
    """Driver code."""

    print('Welcome')
    print('You are about to train a beast of a model, enjoy :)')
    print('Training task: {}'.format(args.task))
    print('Training on the following device: {}'.format(select_device(args.gpu)))

    # Define path for saving model.
    model_name = args.model_dir
    model_dir = f'{DATASET_DIR}/{model_name}'

    # Instantiate model trainer.
    model = TrainModel(args, model_dir)
    model.print_config(args)
    model.train()
    model.save(model_name)
    