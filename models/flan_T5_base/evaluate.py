import nltk
import torch
import os
import gdown  # used to download Scorer model
import pandas as pd
from models.flan_T5_base.bart_score import BARTScorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset


def run(args):
    # Load dataset
    dataset = load_dataset("esnli")

    # Convert to Pandas Dataframe for easy manipulation
    full_test_set = pd.DataFrame(dataset["test"], columns=[
                                 'premise', 'hypothesis', 'label', 'explanation_1', 'explanation_2', 'explanation_3'])

    # Conver label to string based on the following mapping:
    # 0: entailment
    # 1: neutral
    # 2: contradiction
    full_test_set['label'] = full_test_set['label'].map(
        {0: 'entailment', 1: 'neutral', 2: 'contradiction'})

    # TEMPORARY: Only keep first 3 rows, for easy testing
    full_test_set = full_test_set.head(3)
    #

    # Format input string according to model type. One with label (for the "nli-explanation" model, and one without label (for the "label" & "label-explanation" model)
    test_input_without_label = 'premise: ' + \
        full_test_set['premise'] + ' hypothesis: ' + full_test_set['hypothesis']
    test_input_with_label = 'premise: ' + \
        full_test_set['premise'] + ' hypothesis: ' + \
        full_test_set['hypothesis'] + ' label: ' + full_test_set['label']
    test_input_without_label = test_input_without_label.tolist()  # Convert to list
    test_input_with_label = test_input_with_label.tolist()  # Convert to list

    # Format target string
    test_target = full_test_set[[
        'explanation_1', 'explanation_2', 'explanation_3']]
    test_target = test_target.values.tolist()  # Convert to list

    # Evaluate models
    print("Evaluating model: rug-nlp-nli/flan-base-nli-explanation")
    model_results_explanation = evaluateModel(
        "rug-nlp-nli/flan-base-nli-explanation", test_input_with_label, test_target)
    print("Evaluating model: rug-nlp-nli/flan-base-nli-label")
    # model_results_label = evaluateModel(
    #     "rug-nlp-nli/flan-base-nli-label", test_input_without_label, test_target)
    print("Evaluating model: rug-nlp-nli/flan-base-nli-label-explanation")
    model_results_label_explanation = evaluateModel(
        "rug-nlp-nli/flan-base-nli-label-explanation", test_input_without_label, test_target)

    print("Concatenating results and saving to a .csv file...")
    # make dataframe with the input and the results
    results = pd.concat([full_test_set, model_results_explanation,
                        model_results_label_explanation], axis=1)
    # save the results to a csv file
    results.to_csv('results.csv', index=False)


def splitPredictions(model_name, predictions):
    # Split in labels and explanations
    # Create empty lists
    generated_labels = [None] * len(predictions)
    generated_explanations = [None] * len(predictions)
    if model_name == "rug-nlp-nli/flan-base-nli-label-explanation":
        # split based on ": " which follows the label, if it exists
        for i in range(len(predictions)):
            split_predictions = predictions[i].split(": ")
            generated_labels[i] = split_predictions[0]
            generated_explanations[i] = split_predictions[1]

    if model_name == "rug-nlp-nli/flan-base-nli-label":
        # output is only the label
        generated_labels = predictions
        generated_explanations = None

    if model_name == "rug-nlp-nli/flan-base-nli-explanation":
        # output is only the explanation
        generated_labels = None
        generated_explanations = predictions

    return generated_labels, generated_explanations


def evaluateModel(model_name, input, target):
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate predictions
    generated_predictions = generatePredictions(model, tokenizer, input)
    # Split predictions in labels and explanations, depending on what the model produced
    generated_labels, generated_explanations = splitPredictions(
        model_name, generated_predictions)

    # First evaluate explanations: 
    if generated_explanations != None:
        # Calculate scores
        neural_results = neuralEvaluationExplanations(generated_predictions, target)
        # We add the model name, so that the results of different models are distinguishable
        neural_results = neural_results.add_prefix(model_name + '_')
        # amber_results = neuralEvaluation(
        #     generated_predictions, target) # for later, when we have AMBER
        # return pd.concat([neural_results, amber_results], axis=1)  # for later, when we have AMBER
        
    # Then evaluate labels:
    if generated_labels != None:
        print("labels exist. TODO: evaluate labels")
    
    return pd.concat([neural_results], axis=1)


def generatePredictions(model, tokenizer, input):
    # Generate predictions
    predictions = []
    for i in range(len(input)):
        input[i] = tokenizer(
            input[i], truncation=True, return_tensors="pt")
        output = model.generate(
            **input[i], num_beams=8, do_sample=True, max_new_tokens=64)
        decoded_output = tokenizer.batch_decode(
            output, skip_special_tokens=True)[0]
        predictions.append(decoded_output)

    return predictions


def neuralEvaluationExplanations(predictions, target):
    # Load BART scorer
    # Download scorer if it doesn't exist in bartScorer/bart.pth
    if not os.path.exists("data/bartScorer"):
        # if the demo_folder directory is not present then create it.
        os.makedirs("data/bartScorer")
    if (not os.path.exists('data/bartScorer/bart.pth')):
        url = 'https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m'
        output = 'data/bartScorer/bart.pth'
        gdown.download(url, output, quiet=False, fuzzy=True)

    bart_scorer = BARTScorer(device="cuda" if torch.cuda.is_available(
    ) else "cpu", checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='data/bartScorer/bart.pth')

    # Calculate scores
    results = bart_scorer.multi_ref_score(
        predictions, target, agg="max", batch_size=4)

    # construct dataframe with results and predictions
    results = pd.DataFrame(results, columns=['neural_score'])
    results['prediction'] = predictions

    return results
