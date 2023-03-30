import argparse

from models.flan_T5_base import train, predict, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="NLI project",
        description="Welcome to the Natural Language Inference program."
    )

    subparsers = parser.add_subparsers(help='commands', title="commands", dest="command")

    # A train command
    train_parser = subparsers.add_parser(
        'train', help='Fine-tune the base-model')
    train_parser.add_argument(
        "--model_dir",
        default="flan-t5-base-nli-explanation-generation",
        help="Name of the model directory. Defaults to \"flan-t5-base-nli-explanation-generation\""
    )
    train_parser.add_argument(
        "--base_model",
        default="google/flan-t5-base",
        help="Name of the base-model. Defaults to \"'google/flan-t5-base'\""
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
        help="Training and evaluation batch size. Defaults to 32"
    )
    train_parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Number of training epochs. Defaults to 3"
    )
    train_parser.add_argument(
        "--eval_steps",
        default=100,
        type=int,
        help="Number of evaluation steps. Defaults to 100"
    )
    train_parser.add_argument(
        "--logging_steps",
        default=20,
        type=int,
        help="Number of logging steps. Defaults to 20"
    )
    train_parser.add_argument(
        "--save_steps",
        default=100,
        type=int,
        help="Number of steps until saving checkpoint. Defaults to 100"
    )
    train_parser.add_argument(
        "--task",
        default=2,
        type=int,
        help="The type of model being trained. 0: (premise, hypothesis) -> label, 1: (premise, hypothesis, label) -> explanation), 2: (premise, hypothesis) -> (label, explanation). Defaults to 2"
    )


    # A predict command
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
        help="The hypothesis string"
    )
    predict_parser.add_argument(
        "--label",
        choices=['entailment', 'neutral', 'contradiction'],
    )
    predict_parser.add_argument(
        "--model_dir",
        default="flan-t5-base-nli-explanation-generation",
        help="Name of the model directory. Defaults to \"flan-t5-base-nli-explanation-generation\""
    )
    predict_parser.add_argument(
        "--task",
        default=2,
        type=int,
        help="The type of model used for inference. 0: (premise, hypothesis) -> label, 1: (premise, hypothesis, label) -> explanation), 2: (premise, hypothesis) -> (label, explanation). Defaults to 2"
    )


    # An evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate', help='Evaluate')
    
    evaluate_parser.add_argument(
        "--eval_type",
        default="both",
        type=str,
        help="'neural' or 'text' evaluation or 'both'. Defaults to 'both'"
    )
    evaluate_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Inference on a GPU device."
    )



    args = parser.parse_args()
    if args.command == 'train':
        train.run(args)
    if args.command == 'predict':
        predict.run(args)
    if args.command == 'evaluate':
        evaluate.run(args)