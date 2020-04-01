"""
Create a fake dataset by simulating from a known 3PL model. This is to sanity
check our models to make sure things are reasonable.
"""

import os
import torch
import numpy as np

from src.config import DATA_DIR
from src.pyro_core.models import (
    irt_model_1pl, 
    irt_model_2pl, 
    irt_model_3pl,
)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--irt-model', type=str, default='3pl',
                        choices=['1pl', '2pl', '3pl'],
                        help='1pl|2pl|3pl (default: 3pl)')
    parser.add_argument('--num-person', type=int, default=1000,
                        help='number of people to amortize over (default: 1000)')
    parser.add_argument('--num-item', type=int, default=100, 
                        help='number of item to consider (default: 100)')
    parser.add_argument('--ability-dim', type=int, default=1,
                        help='number of ability dimensions (default: 1)')
    parser.add_argument('--nonlinear', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cpu')

    OUT_DIR = os.path.join(DATA_DIR, '{}_simulation_{}person_{}item_{}ability'.format(
        args.irt_model, args.num_person, args.num_item, args.ability_dim))
    if args.nonlinear: OUT_DIR = OUT_DIR + '_nonlinear'
    if not os.path.isdir(OUT_DIR): os.makedirs(OUT_DIR)

    if args.irt_model == '1pl':
        irt_model = irt_model_1pl
    elif args.irt_model == '2pl':
        irt_model = irt_model_2pl
    elif args.irt_model == '3pl':
        irt_model = irt_model_3pl
    else:
        raise Exception('irt_model {} not supported'.format(args.irt_model))

    response, ability, item_feat = irt_model(args.ability_dim, args.num_person, args.num_item, device)
    dataset = {'response': response, 'ability': ability, 'item_feat': item_feat}

    print('Saving to {}'.format(OUT_DIR))
    torch.save(dataset, os.path.join(OUT_DIR, 'simulation.pth'))
