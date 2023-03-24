import nltk
import torch
import pandas as pd
from bart_score import BARTScorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

def run(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("rug-nlp-nli/flan-base-nli-explanation")
    model = AutoModelForSeq2SeqLM.from_pretrained("rug-nlp-nli/flan-base-nli-explanation")
    # Load BART scorer
    bart_scorer = BARTScorer(device="cuda" if torch.cuda.is_available() else "cpu", checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='models/bart.pth')
    # Load dataset
    dataset = load_dataset("esnli")
    
    # Convert to Pandas Dataframe for easy manipulation
    test_set = pd.DataFrame(dataset["test"], columns = ['premise', 'hypothesis', 'explanation_1', 'explanation_2', 'explanation_3'])
    
    #TEMPORARY: Only keep first 10 rows, for easy testing
    test_set = test_set.head(10)
    #
    
    test_input = 'premise: ' + test_set['premise'] + '. hypothesis: ' + test_set['hypothesis'] + '.'
    test_target = test_set[['explanation_1', 'explanation_2', 'explanation_3']]   
    
    test_input = test_input.tolist() # Convert to list
    test_target = test_target.values.tolist() # Convert to list
    
    # Generate predictions
    generatedResponses = []
    for i in range(len(test_input)):
        test_input[i] = tokenizer(test_input[i], truncation=True, return_tensors="pt")
        output = model.generate(**test_input[i], num_beams=8, do_sample=True, max_new_tokens=64)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        generatedResponses.append(decoded_output)
        
    # Calculate scores
    
    results = bart_scorer.multi_ref_score(generatedResponses, test_target, agg="max", batch_size=4)
    
    print(results)
    print("average score: ", sum(results)/len(results))

run(None)