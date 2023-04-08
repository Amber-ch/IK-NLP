"""Main program for carrying out training (fine-tuning), prediction and model evaluation."""

import argparse

from src.model import train, predict, evaluate, preprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="NLI project",
        description="Welcome to the Natural Language Inference program."
    )

    # The 4 subparses are: train, predict, evaluate, preprocess
    subparsers = parser.add_subparsers(help='commands', title="commands", dest="command")

    # Train command
    train_parser = subparsers.add_parser(
        'train', help='Fine-tune the base-model')
    train_parser.add_argument(
        "--model_dir",
        default="flan-t5-base-nli-explanation-generation",
        help="Name of the model directory. Defaults to \"flan-t5-base-nli-explanation-generation\"."
    )
    train_parser.add_argument(
        "--base_model",
        default="google/flan-t5-base",
        help="Name of the base-model. Defaults to \"google/flan-t5-base\"."
    )
    train_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Train on a GPU device."
    )
    train_parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Training and evaluation batch size. Defaults to 32."
    )
    train_parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Number of training epochs. Defaults to 3."
    )
    train_parser.add_argument(
        "--eval_steps",
        default=100,
        type=int,
        help="Number of evaluation steps. Defaults to 100."
    )
    train_parser.add_argument(
        "--logging_steps",
        default=20,
        type=int,
        help="Number of logging steps. Defaults to 20."
    )
    train_parser.add_argument(
        "--save_steps",
        default=100,
        type=int,
        help="Number of steps until saving checkpoint. Defaults to 10."
    )
    train_parser.add_argument(
        "--task",
        default=2,
        type=int,
        choices=[0,1,2],
        help="The type of model being trained. 0: (premise, hypothesis) -> label, 1: (premise, hypothesis, label) -> explanation), 2: (premise, hypothesis) -> (label, explanation). Defaults to 2."
    )
    train_parser.add_argument(
        "--subset_size",
        default=1,
        type=float,
        help="The percentage (from 0 to 1) of the dataset that is used during training. Defaults to 1."
    )
    train_parser.add_argument(
        "--custom_dataset",
        type=str,
        help="Name of the custom dataset. Note: It is loaded with load_from_disk() method, so the dataset should be in the PyArrow format."
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        'predict', help='Generate inference label and explanation')
    predict_parser.add_argument(
        "--premise",
        nargs="*",
        required=True,
        help="The premise string."
    )
    predict_parser.add_argument(
        "--hypothesis",
        nargs="*",
        required=True,
        help="The hypothesis string."
    )
    predict_parser.add_argument(
        "--label",
        choices=['entailment', 'neutral', 'contradiction'],
        help="The corresponding label."
    )
    predict_parser.add_argument(
        "--task",
        default=2,
        type=int,
        choices=[0,1,2],
        help="The type of model used for inference. 0: (premise, hypothesis) -> label, 1: (premise, hypothesis, label) -> explanation), 2: (premise, hypothesis) -> (label, explanation). Defaults to 2."
    )
    predict_parser.add_argument(
        "--model_type",
        default="standard",
        type=str,
        help="'standard' for model trained on e-SNLI dataset as provided, 'custom' for dataset cleaned by template matching. Defaults to 'standard'."
    )
    predict_parser.add_argument(
        "--custom_model",
        type=str,
        help="Name of the custom model. Note: It should be located in the 'data' directory."
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate', help='Evaluate')
    evaluate_parser.add_argument(
        '--model_type',
        default="standard",
        choices=['standard', 'custom'],
        help="'standard' for model trained on e-SNLI dataset as provided, 'custom' for dataset cleaned by template matching. Defaults to 'standard'."
    )
    evaluate_parser.add_argument(
        "--eval_type",
        default="both",
        choices=['neural', 'text', 'both'],
        help="'neural' or 'text' evaluation or 'both'. Defaults to 'both'."
    )
    evaluate_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Inference on a GPU device."
    )
    evaluate_parser.add_argument(
        "--subset_size",
        default=1,
        type=float,
        help="The percentage (from 0 to 1) of the dataset that is used during evaluation. Defaults to 1."
    )
    evaluate_parser.add_argument(
        "--custom_label_model",
        type=str,
        help="Name of the custom label model. Note: It should be located in the 'data' directory."
    )
    evaluate_parser.add_argument(
        "--custom_explanation_model",
        type=str,
        help="Name of the custom explanation model. Note: It should be located in the 'data' directory."
    )
    evaluate_parser.add_argument(
        "--custom_label_explanation_model",
        type=str,
        help="Name of the custom label and explanation model. Note: It should be located in the 'data' directory."
    )

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess', help='Preprocess')
    preprocess_parser.add_argument(
        "--dataset_name",
        default="esnli_reduced",
        type=str,
        help="The name of the dataset to be saved after preprocessing. Defaults to 'esnli_reduced'."
    )
    preprocess_parser.add_argument(
        "--distance_cutoff",
        default=13,
        type=int,
        help="The cutoff value for computing the edit distance between the templates, such that all matched examples will be removed. Defaults to 13."
    )

    # Run the appropriate sub-program with arguments
    args = parser.parse_args()
    if args.command == 'train':
        train.run(args)
    if args.command == 'predict':
        predict.run(args)
    if args.command == 'evaluate':
        evaluate.run(args)
    if args.command == 'preprocess':
        preprocess.run(args)