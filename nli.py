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
        help="Fine-tune the base model."
    )
    train_parser.add_argument(
        "--batch_size",
        default=32,
        type=int
    )
    train_parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int
    )
    train_parser.add_argument(
        "--eval_steps",
        default=100,
        type=int
    )
    train_parser.add_argument(
        "--logging_steps",
        default=20,
        type=int
    )
    train_parser.add_argument(
        "--save_steps",
        default=100,
        type=int
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
        "--model_name",
        default="dummy",
        help="Name of the model (e.g. the most recently trained model). Defaults to \"dummy\""
    )
    predict_parser.add_argument(
        "--model_dir",
        default="flan-t5-base-nli-explanation-generation",
        help="Name of the model directory. Defaults to \"flan-t5-base-nli-explanation-generation\""
    )


    args = parser.parse_args()
    if args.command == 'train':
        train.run(args)
    if args.command == 'predict':
        predict.run(args)