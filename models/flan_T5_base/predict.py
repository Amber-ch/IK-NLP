"""Implements the model inference."""

import nltk

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


DATASET_DIR = 'data'


def run(args):
    """Driver code."""

    # Load punkt tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Load model
    if args.task == 0:
        if args.model_type == "standard":
            model_name = 'rug-nlp-nli/flan-base-nli-label'
        else:
            model_name = 'rug-nlp-nli/flan-base-nli-label-custom'
    elif args.task == 1:
        if args.model_type == "standard":
            model_name = 'rug-nlp-nli/flan-base-nli-explanation'
        else:
            model_name = 'rug-nlp-nli/flan-base-nli-explanation-custom'
    elif args.task == 2:
        if args.model_type == "standard":
            model_name = 'rug-nlp-nli/flan-base-nli-label-explanation'
        else:
            model_name = 'rug-nlp-nli/flan-base-nli-label-explanation-custom'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Format input string according to model type
    if args.label is None:
        test_input = f'premise: {args.premise[0]}. hypothesis: {args.hypothesis[0]}.'
    else:
        test_input = f'premise: {args.premise[0]}. hypothesis: {args.hypothesis[0]}. label: {args.label}'

    # Generate prediction
    test_input = tokenizer(test_input, truncation=True, return_tensors="pt")
    output = model.generate(**test_input, num_beams=8,
                            do_sample=True, max_new_tokens=64)
    decoded_output = tokenizer.batch_decode(
        output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

    # Output prediction
    print(predicted_title)
