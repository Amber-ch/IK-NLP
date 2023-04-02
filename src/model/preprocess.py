import datasets

import pandas as pd
import numpy as np

from src.template import template_matching


# Settings
DATASET_DIR = 'data'


def run(args):

    labels_dict = {
        0 : "entailment",
        1 : "neutral",
        2 : "contradiction"
    }

    esnli = datasets.load_dataset("esnli")
    esnli_train_df = pd.DataFrame(esnli['train'])
    esnli_train_df['label'] = esnli_train_df['label'].map(labels_dict)

    print(f'using cut-off value: {args.distance_cutoff}')
    stats_training_data = template_matching(esnli_train_df['premise'], esnli_train_df['hypothesis'], esnli_train_df['label'], esnli_train_df['explanation_1'], cutoff=args.distance_cutoff)

    # gather all the locations of examples that are to be removed.
    total_indices = []
    total_indices += sum(stats_training_data['indices']['general'].values(), [])
    total_indices += sum(stats_training_data['indices']['entailment'].values(), [])
    total_indices += sum(stats_training_data['indices']['neutral'].values(), [])
    total_indices += sum(stats_training_data['indices']['contradiction'].values(), [])
    total_indices = set(total_indices)

    # update the training split
    esnli['train'] = esnli['train'].select(
        (
            i for i in range(len(esnli['train'])) 
            if i not in total_indices
        )
    )

    print('removed {} examples: {:.2f}% of the train split'.format(len(total_indices), len(total_indices)/esnli_train_df['label'].shape[0] * 100))
    esnli.save_to_disk(f'{DATASET_DIR}/{args.dataset_name}')
    print('done')