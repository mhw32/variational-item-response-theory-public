"""
Do inference with IRT model using HMC on a set of observed data.
"""

import os
import time
import math
import warnings
from tqdm import tqdm

import numpy as np
from sklearn.neighbors.kde import KernelDensity

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.distributions import Normal, Bernoulli
from pyro.infer.mcmc.util import initialize_model, predictive
from pyro.distributions.util import scalar_like, sum_rightmost
from pyro.poutine.util import site_is_subsample
from pyro.util import ignore_experimental_warning

from src.pyro_core.models import (
    irt_model_1pl, 
    irt_model_2pl, 
    irt_model_3pl,
    irt_model_3pl_hierarchical,
)
from src.datasets import (
    load_dataset, 
    wide_to_long_form,
    artificially_mask_dataset,
)
from src.config import OUT_DIR, IS_REAL_WORLD


def sample_posterior_predictive(model, posterior_samples, *args):
    with ignore_experimental_warning():
        predict = predictive(model, posterior_samples, *args)
        return predict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--irt-model', type=str, default='1pl',
                        choices=['1pl', '2pl', '3pl'],
                        help='1pl|2pl|3pl (default: 3pl)')
    parser.add_argument('--dataset', type=str, default='1pl_simulation',
                        choices=[
                            '1pl_simulation', 
                            '2pl_simulation', 
                            '3pl_simulation',
                            'children_language',
                            'duolingo_language',
                            'wordbank_language',
                            'pisa2015_science',
                        ],
                        help='which dataset to run on (default: 1pl_simulation)')
    parser.add_argument('--ability-dim', type=int, default=1,
                        help='number of ability dimensions in data and model (default: 1)')
    parser.add_argument('--num-person', type=int, default=1000,
                        help='number of people in data (default: 1000)')
    parser.add_argument('--num-item', type=int, default=100,
                        help='number of people in data (default: 100)')
    parser.add_argument('--nonlinear', action='store_true', default=False)
    parser.add_argument('--hierarchical', action='store_true', default=False)
    parser.add_argument('--max-num-person', help='limit the number of persons in dataset')
    parser.add_argument('--max-num-item', help='limit the number of items in dataset')
    parser.add_argument('--artificial-missing-perc', type=float, default=0.,
                        help='how much to blank out so we can measure acc (default: 0)')

    parser.add_argument('--num-chains', type=int, default=1)
    parser.add_argument('--num-warmup', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=200)

    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help='where to dump checkpoints (default: OUT_DIR)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pyro.set_rng_seed(args.seed)

    if IS_REAL_WORLD[args.dataset]:
        # these params are only for IRT simulation datasets
        args.num_person = None
        args.num_item = None

        if args.max_num_person is not None:
            args.max_num_person = int(args.max_num_person)
        
        if args.max_num_item is not None:
            args.max_num_item = int(args.max_num_item)

    else:
        args.max_num_person = None
        args.max_num_item = None

    # work around the error "CUDA error: initialization error" when arg.cuda is False
    # see https://github.com/pytorch/pytorch/issues/2517
    torch.multiprocessing.set_start_method("spawn")
    # Enable validation checks
    pyro.enable_validation(__debug__)

    # work around with the error "RuntimeError: received 0 items of ancdata"
    # see https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823
    torch.multiprocessing.set_sharing_strategy("file_system")

    args.out_dir = os.path.join(
        args.out_dir, 
        'hmc_{}_{}_{}person_{}item_{}maxperson_{}maxitem_{}ability_{}maskperc_seed{}'.format(
            args.irt_model, 
            args.dataset, 
            args.num_person, 
            args.num_item, 
            args.max_num_person,
            args.max_num_item,
            args.ability_dim,
            args.artificial_missing_perc,
            args.seed,
        ),
    )
    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    device = torch.device("cuda" if args.cuda else "cpu")

    train_dataset = load_dataset(
        args.dataset, 
        train = True, 
        num_person = args.num_person, 
        num_item = args.num_item, 
        ability_dim = args.ability_dim, 
        nonlinear = args.nonlinear,
        max_num_person = args.max_num_person,
        max_num_item = args.max_num_item,
    )
    num_item = train_dataset.num_item
    num_person = train_dataset.num_person
   
    if args.artificial_missing_perc > 0:
        train_dataset = artificially_mask_dataset(
            train_dataset,
            args.artificial_missing_perc,
        ) 

    response, mask = train_dataset.response, train_dataset.mask
    response[response == -1] = 0  # filler value within support
    response = torch.from_numpy(response).float().to(device)
    mask = torch.from_numpy(mask).long().to(device)

    if len(response.size()) == 2:
        response = response.unsqueeze(2)

    if len(mask.size()) == 2:
        mask = mask.unsqueeze(2)

    if args.irt_model == '1pl':
        irt_model = irt_model_1pl
    elif args.irt_model == '2pl':
        irt_model = irt_model_2pl
    elif args.irt_model == '3pl':
        if args.hierarchical:
            irt_model = irt_model_3pl_hierarchical
        else:
            irt_model = irt_model_3pl
    else:
        raise Exception('irt_model {} not supported.'.format(args.irt_model))

    init_params, potential_fn, transforms, _ = initialize_model(
        irt_model, 
        model_args=(
            args.ability_dim, 
            num_person, 
            num_item, 
            device, 
            response, 
            mask, 
            1,
        ),
        num_chains=args.num_chains,
    )

    start_time = time.time()

    nuts_kernel = NUTS(potential_fn = potential_fn)
    mcmc = MCMC(
        nuts_kernel,
        num_samples = args.num_samples,
        warmup_steps = args.num_warmup,
        num_chains = args.num_chains,
        initial_params = init_params,
        transforms = transforms,
    )
    mcmc.run(
        args.ability_dim, 
        num_person, 
        num_item, 
        device, 
        response, 
        mask, 
        1,
    )
    samples = mcmc.get_samples()
    for key in samples.keys():
        samples[key] = samples[key].cpu()

    end_time = time.time()

    sample_means, sample_variances = {}, {}
    for key, sample in samples.items():
        sample_means[key] = torch.mean(samples[key], dim=0)
        sample_variances[key] = torch.var(samples[key], dim=0)

    # get posterior predictive samples
    posterior_predict_samples = sample_posterior_predictive(
        irt_model, 
        samples, 
        args.ability_dim, 
        num_person, 
        num_item, 
        device,
        None, 
        None, 
        1,
    )

    missing_imputation_accuracy = None
    if args.artificial_missing_perc > 0:
        missing_indices = train_dataset.missing_indices
        missing_labels = train_dataset.missing_labels
        if np.ndim(missing_labels) == 1:
            missing_labels = missing_labels[:, np.newaxis]

        inferred_response = posterior_predict_samples['response'].mean(0)
        inferred_response = torch.round(inferred_response)

        correct, count = 0, 0
        for missing_index, missing_label in zip(missing_indices, missing_labels):
            inferred_label = inferred_response[missing_index[0], missing_index[1]]
            if inferred_label.item() == missing_label[0]:
                correct += 1
            count += 1
        missing_imputation_accuracy = correct / float(count)

    torch.save({
        'samples': samples,
        'sample_means': sample_means,
        'sample_variances': sample_variances,
        'time_elapsed': end_time - start_time,
        'posterior_predict_samples': posterior_predict_samples,
        'missing_imputation_accuracy': missing_imputation_accuracy,
        'args': args,
    }, os.path.join(args.out_dir, 'checkpoint.pth.tar'))
