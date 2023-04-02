import os
import numpy as np
from tqdm import tqdm
import gdown  # used to download Scorer model
import pandas as pd
from models.flan_T5_base.bart_score import BARTScorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric

from models.utils import *

def run(args):
    '''Run evaluation on the test set. 
    
    Args: 
        args (args): The arguments from argparse.
    '''
    eval_type = args.eval_type

    model_suffix = ""
    if args.model_type == "custom":
        model_suffix = "-custom"

    # Make sure a results folder exists
    if not os.path.exists(f"results{model_suffix}"):
        os.makedirs(f"results{model_suffix}")

    # Load dataset
    dataset = load_dataset("esnli")

    # Convert to Pandas Dataframe for easy manipulation
    full_test_set = pd.DataFrame(dataset["test"], columns=[
                                 'premise', 'hypothesis', 'label', 'explanation_1', 'explanation_2', 'explanation_3'])

    # Convert labels to string based on the mapping.
    full_test_set['label'] = full_test_set['label'].map(
        {0: 'entailment', 1: 'neutral', 2: 'contradiction'})

    # For testing: Only keep first 10 rows, for easy testing
    full_test_set = full_test_set.head(5)
    

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
    print(f"Evaluating model: rug-nlp-nli/flan-base-nli-explanation{model_suffix}")
    model_results_explanation = evaluateModel(
        f"rug-nlp-nli/flan-base-nli-explanation{model_suffix}", input_nli_explanation, test_target_explanations, test_target_labels, args)

    print(f"Evaluating model: rug-nlp-nli/flan-base-nli-label{model_suffix}")
    model_results_label = evaluateModel(
        f"rug-nlp-nli/flan-base-nli-label{model_suffix}", input_nli_label, test_target_explanations, test_target_labels, args)

    print(f"Evaluating model: rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}")
    model_results_label_explanation = evaluateModel(
        f"rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}", input_nli_label_explanation, test_target_explanations, test_target_labels, args)

    print("Exporting results to .csv file...")
    # Make dataframe with the input and the results
    results = pd.concat([full_test_set, model_results_explanation, model_results_label,
                        model_results_label_explanation], axis=1)

    # Order in increasing order of 'rug-nlp-nli/flan-base-nli-label-explanation_neural_score', so that the worst results are at the top
    if eval_type in ['neural', 'both']:
        results = results.sort_values(
            by=f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_neural_score', ascending=True)

    # Save the results to a .csv file
    results.to_csv(f'results/allResults{model_suffix}.csv', index=False)

    # Generate summary of results
    print("Generating summary of results...")
    summary = generateSummary(results, args)

    # Save summary to .txt file
    with open(f'results/summary{model_suffix}.txt', 'w') as f:
        f.write(summary)

    print("Evaluation done!")

def clean_df(df, args):
    '''Clean the dataframe by removing duplicate columns and columns that are not needed for the evaluation.
    
    Args: 
        df (pd.DataFrame): The dataframe to clean.
        args (args): The arguments from argparse.
        
    Returns:
        df (pd.DataFrame): The cleaned dataframe.
    '''
    
    model_suffix = ""
    if args.model_type == "custom":
        model_suffix = "-custom"

    # Remove duplicate columns
    df = df.loc[:,~df.columns.duplicated
                
    eval = ["rouge_1", "rouge_2", "rouge_L"]

    # Remove exploded columns
    drop_cols = []
    for i in range(1,4):
        for metric in eval:
            drop_cols.append(f"rug-nlp-nli/flan-base-nli-explanation{model_suffix}_{metric}_{i}")
            drop_cols.append(f"rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_{metric}_{i}")

    # Remove list of explanations column
    drop_cols.append(f"rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_labels")
    drop_cols.append(f"rug-nlp-nli/flan-base-nli-explanation{model_suffix}_labels")

    # Remove duplicate prediction column
    drop_cols.append(f"rug-nlp-nli/flan-base-nli-explanation{model_suffix}_prediction.1")
    drop_cols.append(f"rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_prediction.1")


    df = df.drop(columns = drop_cols)

    return df
    
def evaluateModel(model_name, input, target_explanations, target_labels, args):
    '''Evaluate a model on the given input and target explanations and labels.
    
    Args: 
        model_name (str): The name of the model to evaluate.
        input (list): The input to the model.
        target_explanations (list): The target explanations (3 expanations in the e-SNLI case).
        target_labels (list): The target labels (contradiction, neutral, entailment).
        args (args): The arguments from argparse.
        
    Returns:
        results (pd.DataFrame): All the results (neural, textual & label) of the evaluation of the model with the given name.
    '''
    # select appropriate device
    device = select_device(args.gpu)

    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_type = args.eval_type

    # Generate predictions
    print("Generating predictions for model " + model_name + "...")
    generated_predictions = generatePredictions(model, tokenizer, input, device)
    # Split predictions in labels and explanations, depending on what the model produced
    generated_labels, generated_explanations = splitPredictions(
        model_name, generated_predictions, args)

    # create empty dataframe
    results = pd.DataFrame()
    # First evaluate explanations:
    if generated_explanations != None:
        if eval_type in ['neural', 'both']:
            # Calculate scores
            neural_results = neuralEvaluationExplanations(model_name,
                                                          generated_predictions, target_explanations, device)
            neural_results = neural_results.add_prefix(model_name + '_')
            results = pd.concat([results, neural_results], axis=1)
        if eval_type in ['text', 'both']:
            # We add the model name, so that the results of different models are distinguishable
            text_results = textEvaluationExplanations(
                model_name, generated_predictions, target_explanations)  # for later, when we have AMBER
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
    '''Evaluate the labels of the model.
    
    Args: 
        output (list): The output labels of the model.
        target (list): The target labels.
        
    Returns:
        label_results (pd.DataFrame): A dataframe with the results of the label evaluation (whether each label is correct & difference).
    '''
    # Evaluates whether the model predicted the correct label. Returns a dataframe with the results.
    label_results = pd.DataFrame()
    # for each output, check if it is equal to the target
    for i in range(len(output)):
        label_results.at[i, 'correct_label?'] = (output[i] == target[i])
        label_results.at[i, 'label_difference'] = " predicted: " + \
            output[i] + ", target: " + target[i]
    return label_results

def generatePredictions(model, tokenizer, input, device):
    '''Generate predictions for the given input.
    
    Args: 
        model (transformers.modeling_bart.BartForConditionalGeneration): The model to generate predictions with.
        tokenizer (transformers.tokenization_bart.BartTokenizer): The tokenizer to use for the model.
        input (list): The input to the model.
        device (torch.device): The device to use for the model.
        
    Returns:
        predictions (list): The predictions made by the model (list of strings).
    '''
    # Generate predictions
    predictions = []
    for i in tqdm(range(len(input))):
        input[i] = tokenizer(
            input[i], truncation=True, return_tensors="pt").to(device)
        output = model.generate(
            **input[i], num_beams=8, do_sample=True, max_new_tokens=64)
        decoded_output = tokenizer.batch_decode(
            output, skip_special_tokens=True)[0]
        predictions.append(decoded_output)

    return predictions

def neuralEvaluationExplanations(model_name, predictions, target, device):
    '''Evaluate the explanations of the model using Bart Scorer.
        
    Args:
        model_name (str): The name of the model to evaluate.
        predictions (list): The predictions made by the model.
        target (list): The target explanations (three explanations per input).
        device (torch.device): The device to use for the model.
        
    Returns:
        results (pd.DataFrame): A dataframe with the results of the neural evaluation (based on BART score).
    '''
    
    # Load BART scorer
    # Download scorer if it doesn't exist in bartScorer/bart.pth
    if not os.path.exists("data/bartScorer"):
        # if the demo_folder directory is not present then create it.
        os.makedirs("data/bartScorer")
    if (not os.path.exists('data/bartScorer/bart.pth')):
        url = 'https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m'
        output = 'data/bartScorer/bart.pth'
        gdown.download(url, output, quiet=False, fuzzy=True)

    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='data/bartScorer/bart.pth')

    # Calculate scores
    results = bart_scorer.multi_ref_score(
        predictions, target, agg="max", batch_size=4)

    # Apply np.exp() to column, so that the scores are between 0 and 1
    results = np.exp(results)
    # Construct dataframe with results and predictions
    results = pd.DataFrame(results, columns=['neural_score'])
    results['prediction'] = predictions

    results.to_csv(f'results/{model_name[12:]}_neural_scores.csv')  # save to csv file.
    return results

def textEvaluationExplanations(model_name, predictions, target):
    '''Evaluate the explanations of the model using text-based metrics.
        
    Args:
        model_name (str): The name of the model to evaluate.
        predictions (list): The predictions made by the model.
        target (list): The target explanations (three explanations per input).
        
    Returns:
        results (pd.DataFrame): A dataframe with the results of the text-based evaluation (based on ROUGE scores).
    '''
    # Calculate scores
    scores_labels_predictions_df = compute_metrics(predictions, target)
    
    # Construct dataframe with results and predictions
    rouge_scores_explode = scores_labels_predictions_df

    # Explode the scores and labels
    rouge_1_columns = ['rouge_1_1', 'rouge_1_2', 'rouge_1_3']
    rouge_2_columns = ['rouge_2_1', 'rouge_2_2', 'rouge_2_3']
    rouge_L_columns = ['rouge_L_1', 'rouge_L_2', 'rouge_L_3']

    # Add columns for the scores and labels
    rouge_scores_explode[['label_1', 'label_2', 'label_3']] = pd.DataFrame(
        scores_labels_predictions_df.labels.tolist(), index=scores_labels_predictions_df.index)
    rouge_scores_explode[rouge_1_columns] = pd.DataFrame(
        scores_labels_predictions_df.rouge_1.tolist(), index=scores_labels_predictions_df.index)
    rouge_scores_explode[rouge_2_columns] = pd.DataFrame(
        scores_labels_predictions_df.rouge_2.tolist(), index=scores_labels_predictions_df.index)
    rouge_scores_explode[rouge_L_columns] = pd.DataFrame(
        scores_labels_predictions_df.rouge_L.tolist(), index=scores_labels_predictions_df.index)
    
    # Explode the scores and labels
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
    
    # Calculate max rouge scores
    rouge_scores_explode['rouge_1_max'] = rouge_1_df.max(axis=1)
    rouge_scores_explode['rouge_2_max'] = rouge_2_df.max(axis=1)
    rouge_scores_explode['rouge_L_max'] = rouge_L_df.max(axis=1)

    # Save results to csv file
    rouge_scores_explode.to_csv(f'results/{model_name[12:]}_rouge_scores.csv')

    return rouge_scores_explode

def generateSummary(results, args):
    '''Generate a summary of the results.
        
    Args:
        results (pd.DataFrame): A dataframe with the results of the evaluation. 
        args (args): The arguments from argparse.
        
    Returns:
        summary (str): A summary of the results.
    '''
    
    summary = ""
    eval_type = args.eval_type

    model_suffix = ""
    # If the model is a custom model, add the suffix "-custom" to the model name
    if args.model_type == "custom":
        model_suffix = "-custom"


    if eval_type in ['neural', 'both']:
        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_neural_score
        mean = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_neural_score'].mean(), 2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_neural_score'].std(), 2)
        summary += f"Average explanation neural score of nli-explanation{model_suffix}: " + \
            str(mean) + " (std: " + str(std) + ")\n"

        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-label_neural_score
        mean = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_neural_score'].mean(), 2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_neural_score'].std(), 2)
        summary += f"Average explanation neural score of nli-label-explanation{model_suffix}: " + \
            str(mean) + " (std: " + str(std) + ")\n"

    if eval_type in ['text', 'both']:
        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_rouge_1_max
        mean = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_1_max'].mean(), 2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_1_max'].std(), 2)
        summary += f"Average explanation text score of rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_1_max: " + \
            str(mean) + " (std: " + str(std) + ")\n"

        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_rouge_2_max
        mean = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_2_max'].mean(), 2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_2_max'].std(), 2)
        summary += f"Average explanation text score of rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_2_max: " + \
            str(mean) + " (std: " + str(std) + ")\n"

        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_rouge_L_max
        mean = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_L_max'].mean(),2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_L_max'].std(),2)
        summary += f"Average explanation text score of rug-nlp-nli/flan-base-nli-explanation{model_suffix}_rouge_L_max: " + \
            str(mean) + " (std: " + str(std) + ")\n"

        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_rouge_1_max
        mean = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_1_max'].mean(),2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_1_max'].std(),2)
        summary += f"Average explanation text score of rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_1_max: " + \
            str(mean) + " (std: " + str(std) + ")\n"

        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_rouge_2_max
        mean = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_2_max'].mean(),2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_2_max'].std(),2)
        summary += f"Average explanation text score of rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_2_max: " + \
            str(mean) + " (std: " + str(std) + ")\n"

        # Find average and standard deviation of column rug-nlp-nli/flan-base-nli-explanation_rouge_L_max
        mean = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_L_max'].mean(),2)
        std = round(results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_L_max'].std(),2)
        summary += f"Average explanation text score of rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_rouge_L_max: " + \
            str(mean) + " (std: " + str(std) + ")\n"

    # Find percentage of correct labels of nli-label
    correct_labels = results[f'rug-nlp-nli/flan-base-nli-label{model_suffix}_correct_label?'].sum()
    total_labels = len(
        results[f'rug-nlp-nli/flan-base-nli-label{model_suffix}_correct_label?'])
    percentage = str(round(correct_labels/total_labels, 2))
    summary += f"Percentage of correct labels of nli-label{model_suffix}: " + percentage + "\n"

    # Find percentage of correct labels of nli-label-explanation
    correct_labels = results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_correct_label?'].sum()
    total_labels = len(
        results[f'rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}_correct_label?'])
    percentage = str(round(correct_labels/total_labels, 2))
    summary += f"Percentage of correct labels of nli-label-explanation{model_suffix}: " + percentage + "\n"

    return summary

def splitPredictions(model_name, predictions, args):
    '''Split the predictions into labels and explanations
    
    Args:
        model_name (str): name of the model. Used to determine what split is needed
        predictions (list): list of predictions that need to be split
        args (args): The arguments from argparse.
    
    Returns:
        list: list of labels and list of explanations. Either is None if the model does not output that.
    '''
    model_suffix = ""
    if args.model_type == "custom":
        model_suffix = "-custom"

    # Split in labels and explanations
    # Create empty lists
    generated_labels = [None] * len(predictions)
    generated_explanations = [None] * len(predictions)
    if model_name == f"rug-nlp-nli/flan-base-nli-label-explanation{model_suffix}":
        # split based on ": " which follows the label, if it exists
        for i in range(len(predictions)):
            split_predictions = predictions[i].split(": ")
            generated_labels[i] = split_predictions[0]
            generated_explanations[i] = split_predictions[1]

    if model_name == f"rug-nlp-nli/flan-base-nli-label{model_suffix}":
        # output is only the label
        generated_labels = predictions
        generated_explanations = None

    if model_name == f"rug-nlp-nli/flan-base-nli-explanation{model_suffix}":
        # output is only the explanation
        generated_labels = None
        generated_explanations = predictions

    return generated_labels, generated_explanations

def compute_metrics(predictions, targets):
    '''Compute the ROUGE scores for the predictions and targets

    Args:
        predictions (list): list of generated predictions
        targets (list): list of targets
    
    Returns:
        scores_labels_predictions (pd.DataFrame): dataframe with the predictions, targets and the ROUGE scores
    '''

    metric = load_metric("rouge")

    # Compute ROUGE scores
    result = metric.compute(predictions=predictions, references=targets,
                            use_stemmer=True)
    rouge_1 = []
    rouge_2 = []
    rouge_L = []
    for i in range(len(predictions)):
        result = [metric.compute(predictions=[
                                 predictions[i]]*3, references=targets[i], use_stemmer=True, use_aggregator=False)]

        rouge_1.append(result[0]['rouge1'])
        rouge_2.append(result[0]['rouge2'])
        rouge_L.append(result[0]['rougeL'])
        
    # Create dataframe with predictions, targets and ROUGE scores
    scores_labels_predictions = pd.DataFrame(
        {"prediction": predictions, "labels": targets, "rouge_1": rouge_1, "rouge_2": rouge_2, "rouge_L": rouge_L})

    return scores_labels_predictions
