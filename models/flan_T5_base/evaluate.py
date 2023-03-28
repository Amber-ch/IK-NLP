import os
import numpy as np
import nltk
import re
import torch
import gdown  # used to download Scorer model
import pandas as pd
from models.flan_T5_base.bart_score import BARTScorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric


def run(args):
    # Load dataset
    dataset = load_dataset("esnli")

    # Convert to Pandas Dataframe for easy manipulation
    full_test_set = pd.DataFrame(dataset["test"], columns=[
                                 'premise', 'hypothesis', 'label', 'explanation_1', 'explanation_2', 'explanation_3'])

    # Convert labels to string based on the mapping.
    full_test_set['label'] = full_test_set['label'].map(
        {0: 'entailment', 1: 'neutral', 2: 'contradiction'})

    # TEMPORARY: Only keep first 1 rows, for easy testing
    full_test_set = full_test_set.head(1)
    #

    # Format input string according to model type. One with label (for the "nli-explanation" model, and one without label (for the "label" & "label-explanation" model)

    input_nli_explanation = 'premise: ' + \
        full_test_set['premise'] + ' hypothesis: ' + \
        full_test_set['hypothesis'] + ' label: ' + full_test_set['label']
    input_nli_label = 'premise: ' + \
        full_test_set['premise'] + ' hypothesis: ' + \
        full_test_set['hypothesis']
    input_nli_label_explanation = 'premise: ' + \
        full_test_set['premise'] + ' hypothesis: ' + \
        full_test_set['hypothesis']

    input_nli_explanation = input_nli_explanation.tolist()  # Convert to list
    input_nli_label = input_nli_label.tolist()  # Convert to list
    input_nli_label_explanation = input_nli_label_explanation.tolist()  # Convert to list

    # Format target explanations
    test_target_explanations = full_test_set[[
        'explanation_1', 'explanation_2', 'explanation_3']]
    test_target_explanations = test_target_explanations.values.tolist()  # Convert to list

    # Format target labels
    test_target_labels = full_test_set['label'].tolist()  # Convert to list

    # Evaluate models
    print("Evaluating model: rug-nlp-nli/flan-base-nli-explanation")
    model_results_explanation = evaluateModel(
        "rug-nlp-nli/flan-base-nli-explanation", input_nli_explanation, test_target_explanations, test_target_labels, args)

    print("Evaluating model: rug-nlp-nli/flan-base-nli-label")
    model_results_label = evaluateModel(
        "rug-nlp-nli/flan-base-nli-label", input_nli_label, test_target_explanations, test_target_labels, args)

    print("Evaluating model: rug-nlp-nli/flan-base-nli-label-explanation")
    model_results_label_explanation = evaluateModel(
        "rug-nlp-nli/flan-base-nli-label-explanation", input_nli_label_explanation, test_target_explanations, test_target_labels, args)

    print("Exporting results to .csv file...")
    # Make dataframe with the input and the results
    results = pd.concat([full_test_set, model_results_explanation, model_results_label,
                        model_results_label_explanation], axis=1)

    # Order in increasing order of 'rug-nlp-nli/flan-base-nli-label-explanation_neural_score', so that the worst results are at the top
    results = results.sort_values(
        by='rug-nlp-nli/flan-base-nli-label-explanation_neural_score', ascending=True)

    # Save the results to a .csv file
    results.to_csv('results.csv', index=False)

    # Generate summary of results
    print("Generating summary of results...")
    summary = generateSummary(results)

    # Save summary to .txt file
    with open('summary.txt', 'w') as f:
        f.write(summary)

    print("Evaluation done!")


def evaluateModel(model_name, input, target_explanations, target_labels, args):
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_type = args.eval_type
    
    # Generate predictions
    generated_predictions = generatePredictions(model, tokenizer, input)
    # Split predictions in labels and explanations, depending on what the model produced
    generated_labels, generated_explanations = splitPredictions(
        model_name, generated_predictions)

    # create empty dataframe
    results = pd.DataFrame()
    # First evaluate explanations:
    if generated_explanations != None:
        if eval_type in ['neural', 'both']:
        # Calculate scores
            neural_results = neuralEvaluationExplanations(
            generated_predictions, target_explanations)            
            neural_results = neural_results.add_prefix(model_name + '_')
            results = pd.concat([results, neural_results], axis=1)
        if eval_type in ['text', 'both']:
        # We add the model name, so that the results of different models are distinguishable
            text_results = textEvaluationExplanations(generated_predictions, target_explanations) # for later, when we have AMBER
            text_results = text_results.add_prefix(model_name + '_')
            results = pd.concat([results, text_results], axis=1)        

    # Then evaluate labels:
    if generated_labels != None:
        # Evaluate labels
        label_results = evaluateLabels(generated_labels, target_labels)
        label_results = label_results.add_prefix(model_name + '_')
        results = pd.concat([results, label_results], axis=1)

    return results


def evaluateLabels(output, target):
    # Evaluates whether the model predicted the correct label. Returns a dataframe with the results.
    label_results = pd.DataFrame()
    # for each output, check if it is equal to the target
    for i in range(len(output)):
        label_results.at[i, 'correct_label?'] = (output[i] == target[i])
        label_results.at[i, 'label_difference'] = " predicted: " + \
            output[i] + ", target: " + target[i]
    return label_results


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
    # Apply np.exp() to column, so that the scores are between 0 and 1
    results = np.exp(results) 
    # Construct dataframe with results and predictions
    results = pd.DataFrame(results, columns=['neural_score'])
    results['prediction'] = predictions
    return results

def textEvaluationExplanations(predictions, target):
    metric = load_metric("rouge")
    scores_labels_predictions_df = compute_metrics(predictions, target)

    rouge_scores_explode = scores_labels_predictions_df

    rouge_1_columns = ['rouge_1_1', 'rouge_1_2', 'rouge_1_3']
    rouge_2_columns = ['rouge_2_1', 'rouge_2_2', 'rouge_2_3']
    rouge_L_columns = ['rouge_L_1', 'rouge_L_2', 'rouge_L_3']

    rouge_scores_explode[['label_1', 'label_2', 'label_3']] = pd.DataFrame(scores_labels_predictions_df.labels.tolist(), index=scores_labels_predictions_df.index)
    rouge_scores_explode[rouge_1_columns] = pd.DataFrame(scores_labels_predictions_df.rouge_1.tolist(), index=scores_labels_predictions_df.index)
    rouge_scores_explode[rouge_2_columns] = pd.DataFrame(scores_labels_predictions_df.rouge_2.tolist(), index=scores_labels_predictions_df.index)
    rouge_scores_explode[rouge_L_columns] = pd.DataFrame(scores_labels_predictions_df.rouge_L.tolist(), index=scores_labels_predictions_df.index)

    f_rouge_1_1 = rouge_scores_explode['rouge_1_1'].apply(lambda x: x.fmeasure) 
    f_rouge_1_2 = rouge_scores_explode['rouge_1_2'].apply(lambda x: x.fmeasure) 
    f_rouge_1_3 = rouge_scores_explode['rouge_1_3'].apply(lambda x: x.fmeasure) 
    rouge_1_df = pd.DataFrame(data=[f_rouge_1_1, f_rouge_1_2, f_rouge_1_3]).T

    f_rouge_2_1 = rouge_scores_explode['rouge_2_1'].apply(lambda x: x.fmeasure) 
    f_rouge_2_2 = rouge_scores_explode['rouge_2_2'].apply(lambda x: x.fmeasure) 
    f_rouge_2_3 = rouge_scores_explode['rouge_2_3'].apply(lambda x: x.fmeasure) 
    rouge_2_df = pd.DataFrame(data=[f_rouge_2_1, f_rouge_2_2, f_rouge_2_3]).T

    f_rouge_L_1 = rouge_scores_explode['rouge_L_1'].apply(lambda x: x.fmeasure) 
    f_rouge_L_2 = rouge_scores_explode['rouge_L_2'].apply(lambda x: x.fmeasure) 
    f_rouge_L_3 = rouge_scores_explode['rouge_L_3'].apply(lambda x: x.fmeasure) 
    rouge_L_df = pd.DataFrame(data=[f_rouge_L_1, f_rouge_L_2, f_rouge_L_3]).T

    rouge_scores_explode['rouge_1_max'] = rouge_1_df.max(axis=1)
    rouge_scores_explode['rouge_2_max'] = rouge_2_df.max(axis=1)
    rouge_scores_explode['rouge_L_max'] = rouge_L_df.max(axis=1)

    rouge_scores_explode.to_csv('rouge_scores.csv')

    for metric in ['rouge_1_max', 'rouge_2_max', 'rouge_L_max']:
        print('mean ' , metric, ': ', np.mean(rouge_scores_explode[metric]))
        print('stdev ', metric, ': ', np.std(rouge_scores_explode[metric]))
        print()

    return rouge_scores_explode

def generateSummary(results):
    summary = ""
    # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_neural_score
    mean = results['rug-nlp-nli/flan-base-nli-explanation_neural_score'].mean().round(2)
    std = results['rug-nlp-nli/flan-base-nli-explanation_neural_score'].std().round(2)
    summary += "Average explanation score of nli-explanation: " + \
        str(mean) + " (std: " + str(std) + ")\n"

    # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-label_neural_score
    mean = results['rug-nlp-nli/flan-base-nli-label-explanation_neural_score'].mean().round(2)
    std = results['rug-nlp-nli/flan-base-nli-label-explanation_neural_score'].std().round(2)
    summary += "Average explanation score of nli-label-explanation: " + \
        str(mean) + " (std: " + str(std) + ")\n"

    # Find percentage of correct labels of nli-label
    correct_labels = results['rug-nlp-nli/flan-base-nli-label_correct_label?'].sum()
    total_labels = len(
        results['rug-nlp-nli/flan-base-nli-label_correct_label?'])
    percentage = str(round(correct_labels/total_labels, 2))
    summary += "Percentage of correct labels of nli-label: " + percentage + "\n"

    # Find percentage of correct labels of nli-label-explanation
    correct_labels = results['rug-nlp-nli/flan-base-nli-label-explanation_correct_label?'].sum()
    total_labels = len(
        results['rug-nlp-nli/flan-base-nli-label-explanation_correct_label?'])
    percentage = str(round(correct_labels/total_labels, 2))
    summary += "Percentage of correct labels of nli-label-explanation: " + percentage + "\n"

    return summary

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

def compute_metrics(predictions, targets):

    metric = load_metric("rouge")

    # Compute ROUGE scores
    result = metric.compute(predictions=predictions, references=targets,
                            use_stemmer=True)
    rouge_1 = []
    rouge_2 = []
    rouge_L = []
    for i in range(len(predictions)):
      result = [metric.compute(predictions=[predictions[i]]*3, references=targets[i], use_stemmer=True, use_aggregator=False)]

      rouge_1.append(result[0]['rouge1'])
      rouge_2.append(result[0]['rouge2'])
      rouge_L.append(result[0]['rougeL'])

    scores_labels_predictions = pd.DataFrame({"prediction": predictions, "labels": targets, "rouge_1":rouge_1, "rouge_2":rouge_2, "rouge_L":rouge_L})

    return scores_labels_predictions