"""Create poster_predictives for VIBO_ models."""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.distributions as dist

from src.torch_core.models import (
    VIBO_1PL, 
    VIBO_2PL, 
    VIBO_3PL,
)
from src.datasets import load_dataset, artificially_mask_dataset
from src.utils import AverageMeter


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--num-posterior-samples', type=int, default=200)
    args = parser.parse_args()
    num_posterior_samples = args.num_posterior_samples

    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    args = checkpoint['args']
    args.num_posterior_samples = num_posterior_samples

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda: torch.cuda.set_device(args.gpu_device)

    if args.irt_model == '1pl':
        model_class = VIBO_1PL
    elif args.irt_model == '2pl':
        model_class = VIBO_2PL
    elif args.irt_model == '3pl':
        model_class = VIBO_3PL
    else:
        raise Exception(f'model {args.irt_model} not recognized')

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

    num_person = train_dataset.num_person
    num_item   = train_dataset.num_item
    
    model = model_class(
        args.ability_dim,
        num_item,
        hidden_dim = args.hidden_dim,
        ability_merge = args.ability_merge,
        conditional_posterior = args.conditional_posterior,
        generative_model = args.generative_model,
    ).to(device)

    def sample_posterior_predictive(loader):
        model.eval()
        meter = AverageMeter()
        pbar = tqdm(total=len(loader))

        with torch.no_grad():
            
            response_sample_set = []

            for _, response, _, mask in loader:
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)

                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                    model.encode(response, mask)
                
                ability_scale = torch.exp(0.5 * ability_logvar)
                item_feat_scale = torch.exp(0.5 * item_feat_logvar)

                ability_posterior = dist.Normal(ability_mu, ability_scale)
                item_feat_posterior = dist.Normal(item_feat_mu, item_feat_scale)
                
                ability_samples = ability_posterior.sample([args.num_posterior_samples])
                item_feat_samples = item_feat_posterior.sample([args.num_posterior_samples])

                response_samples = []
                for i in range(args.num_posterior_samples):
                    ability_i = ability_samples[i]
                    item_feat_i = item_feat_samples[i]
                    response_i = model.decode(ability_i, item_feat_i).cpu()
                    response_samples.append(response_i)
                response_samples = torch.stack(response_samples)
                response_sample_set.append(response_samples)

                pbar.update()

            response_sample_set = torch.cat(response_sample_set, dim=1)

            pbar.close()

        return {'response': response_sample_set}

    model.load_state_dict(state_dict)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size, 
        shuffle = False,
    )

    posterior_predict_samples = sample_posterior_predictive(train_loader)
    checkpoint['posterior_predict_samples'] = posterior_predict_samples

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
        checkpoint['missing_imputation_accuracy'] = missing_imputation_accuracy
        print(missing_imputation_accuracy)

    torch.save(checkpoint, checkpoint_path)
