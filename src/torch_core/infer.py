"""Create infer_dict for VIBO_ models."""

import os
from tqdm import tqdm

import torch
from src.torch_core.models import (
    VIBO_1PL, 
    VIBO_2PL, 
    VIBO_3PL,
)
from src.datasets import load_dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    args = checkpoint['args']

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

    def get_infer_dict(loader):
        model.eval()
        infer_dict = {}

        with torch.no_grad(): 
            ability_mus, item_feat_mus = [], []
            ability_logvars, item_feat_logvars = [], []

            pbar = tqdm(total=len(loader))
            for _, response, _, mask in loader:
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)

                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                    model.encode(response, mask)

                ability_mus.append(ability_mu.cpu())
                ability_logvars.append(ability_logvar.cpu())

                item_feat_mus.append(item_feat_mu.cpu())
                item_feat_logvars.append(item_feat_logvar.cpu())

                pbar.update()

            ability_mus = torch.cat(ability_mus, dim=0)
            ability_logvars = torch.cat(ability_logvars, dim=0)
            pbar.close()

        infer_dict['ability_mu'] = ability_mus
        infer_dict['ability_logvar'] = ability_logvars
        infer_dict['item_feat_mu'] = item_feat_mu
        infer_dict['item_feat_logvar'] = item_feat_logvar

        return infer_dict

    model.load_state_dict(state_dict)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size, 
        shuffle = False,
    )

    infer_dict = get_infer_dict(train_loader)
    checkpoint['infer_dict'] = infer_dict

    torch.save(checkpoint, checkpoint_path)
