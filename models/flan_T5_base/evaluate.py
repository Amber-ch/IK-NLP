from bart_score import BARTScorer

def run(args):
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart.pth')
    bart_scorer.score(['This is interesting.'], ['This is fun.'], batch_size=4)

run(None)