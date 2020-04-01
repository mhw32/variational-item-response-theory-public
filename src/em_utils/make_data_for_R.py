import os
import time
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.distributions as dist
import torch.nn.functional as F

from src.datasets import load_dataset, artificially_mask_dataset
from src.config import R_DATA_DIR


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='1pl_simulation',
                        choices=[
                            '1pl_simulation', 
                            '2pl_simulation', 
                            '3pl_simulation',
                            '1pl_nonlinear',
                            '2pl_nonlinear',
                            '3pl_nonlinear',
                            'children_language',
                            'duolingo_language',
                            'wordbank_language',
                            'pisa2015_science',
                        ],
                        help='which dataset to run on (default: 1pl_simulation)')
    parser.add_argument('--ability-dim', type=int, default=1,
                        help='number of ability dimensions (default: 1)')
    parser.add_argument('--num-person', type=int, default=1000,
                        help='number of people in data (default: 1000)')
    parser.add_argument('--num-item', type=int, default=100,
                        help='number of people in data (default: 100)')
    parser.add_argument('--max-num-person', help='limit the number of persons in dataset')
    parser.add_argument('--max-num-item', help='limit the number of items in dataset')
    parser.add_argument('--artificial-missing-perc', type=float, default=0.,
                        help='how much to blank out so we can measure acc (default: 0)')
    args = parser.parse_args()

    if not os.path.isdir(R_DATA_DIR):
        os.makedirs(R_DATA_DIR)

    data_dir = os.path.join(
        R_DATA_DIR, 
        f'{args.dataset}_{args.num_person}person_{args.num_item}item_{args.max_num_person}maxperson_{args.max_num_item}maxitem_{args.ability_dim}ability',
    )

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    train_dataset = load_dataset(
        args.dataset,
        train = True, 
        num_person = args.num_person, 
        num_item = args.num_item,  
        ability_dim = args.ability_dim,
        max_num_person = args.max_num_person,
        max_num_item = args.max_num_item,
    )

    if args.artificial_missing_perc > 0:
        train_dataset = artificially_mask_dataset(
            train_dataset,
            args.artificial_missing_perc,
        ) 

    response = train_dataset.response
    mask = train_dataset.mask
    missing_indices = train_dataset.missing_indices
    missing_labels = train_dataset.missing_labels

    np.save(os.path.join(data_dir, 'response.npy'), response)
    np.save(os.path.join(data_dir, 'mask.npy'), mask)
    np.save(os.path.join(data_dir, 'missing_indices.npy'), missing_indices)
    np.save(os.path.join(data_dir, 'missing_labels.npy'), missing_labels)
