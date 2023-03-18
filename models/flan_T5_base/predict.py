import nltk

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


DATASET_DIR = 'data'


def run(args):

    # TODO: refactor code
    nltk.download('punkt')

    # Load model
    model_name = f'{args.model_dir}/{args.model_name}'
    model_dir = f'{DATASET_DIR}/{model_name}'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # Format input string
    test_input = f'premise: {args.premise[0]}. hypothesis: {args.hypothesis}[0].'

    # Generate prediction
    test_input = tokenizer(test_input, truncation=True, return_tensors="pt")
    output = model.generate(**test_input, num_beams=8, do_sample=True, max_new_tokens=64)#, min_length=10, max_length=64)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

    # Prints
    print(predicted_title)
