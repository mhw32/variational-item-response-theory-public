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
from src.utils import AverageMeter


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
    test_dataset  = load_dataset(
        args.dataset, 
        train = False, 
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

    def get_log_marginal_density(loader):
        model.eval()
        meter = AverageMeter()
        pbar = tqdm(total=len(loader))

        with torch.no_grad():
            for _, response, _, mask in loader:
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)

                marginal = model.log_marginal(
                    response, 
                    mask, 
                    num_samples = 10,
                )
                marginal = torch.mean(marginal)
                meter.update(marginal.item(), mb)

                pbar.update()
                pbar.set_postfix({'Marginal': meter.avg})
        
        pbar.close()
        print('====> Marginal: {:.4f}'.format(meter.avg))

        return meter.avg

    model.load_state_dict(state_dict)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size, 
        shuffle = False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = args.batch_size, 
        shuffle = False,
        num_workers = args.num_workers,
    )

    train_logp = get_log_marginal_density(train_loader)
    test_logp = get_log_marginal_density(test_loader)

    checkpoint['train_logp'] = train_logp
    checkpoint['test_logp'] = test_logp

    torch.save(checkpoint, checkpoint_path)
