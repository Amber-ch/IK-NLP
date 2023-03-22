import torch
from bart_score import BARTScorer

def run(args):
    bart_scorer = BARTScorer(device="cuda" if torch.cuda.is_available() else "cpu", checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='models/bart.pth')
    
    # We load data here. For now placeholders.
    input = ["I'm super happy today.", "This is a good idea."]
    target = [["I feel good today.", "I feel sad today."], ["Not bad.", "Sounds like a good idea."]]
    
    # We generate the predictions, for now placeholders.
    generatedResponses = ["This is the output for one.", "This is the output for two"]
    
    
    results = bart_scorer.multi_ref_score(generatedResponses, target, agg="max", batch_size=4)
    
    print(results)
    print("average score: ", sum(results)/len(results))

run(None)