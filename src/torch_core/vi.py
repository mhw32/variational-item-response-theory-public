import os
import time
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.distributions as dist
import torch.nn.functional as F

from src.torch_core.models import VI_1PL, VI_2PL, VI_3PL
from src.datasets import load_dataset, artificially_mask_dataset
from src.utils import AverageMeter, save_checkpoint
from src.config import OUT_DIR


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--irt-model', type=str, default='1pl',
                        choices=['1pl', '2pl', '3pl'],
                        help='1pl|2pl|3pl (default: 1pl)')
    parser.add_argument('--dataset', type=str, default='1pl_simulation',
                        choices=[
                            '1pl_simulation', 
                            '2pl_simulation', 
                            '3pl_simulation',
                            'critlangacq',
                            'duolingo',
                            'wordbank',
                            'pisa2015_science',
                        ],
                        help='which dataset to run on (default: 1pl_simulation)')

    parser.add_argument('--ability-dim', type=int, default=1,
                        help='number of ability dimensions (default: 1)')
    parser.add_argument('--artificial-missing-perc', type=float, default=0.,
                        help='how much to blank out so we can measure acc (default: 0)')

    parser.add_argument('--num-person', type=int, default=1000,
                        help='number of people in data (default: 1000)')
    parser.add_argument('--num-item', type=int, default=100,
                        help='number of people in data (default: 100)')
    parser.add_argument('--num-posterior-samples', type=int, default=400,
                        help='number of samples to use for analysis (default: 400)')
    parser.add_argument('--max-num-person', help='limit the number of persons in dataset')
    parser.add_argument('--max-num-item', help='limit the number of items in dataset')

    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help='where to save chkpts (default: OUT_DIR)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='default learning rate: 5e-3')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--max-iters', type=int, default=-1, metavar='N',
                        help='number of maximum iterations (default: -1)')
    parser.add_argument('--anneal-kl', action='store_true', default=False,
                        help='anneal KL divergence (default: False)')
    parser.add_argument('--beta-kl', type=float, default=1.0,
                        help='constant multiplier on KL (default: 1.0)')
    parser.add_argument('--prior-scale', type=float, default=1.,
                        help='standard deviation for prior normal (default: 1.)')
    parser.add_argument('--no-marginal', action='store_true', default=False,
                        help='if true, skip marginal loglike computation (default: False)')
    parser.add_argument('--no-predictive', action='store_true', default=False,
                        help='if true, skip posterior predictive computation (default: False)')
    parser.add_argument('--reduce-lr', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=100, 
                        help='number of epochs before reducing lr (default: 100)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--gpu-device', type=int, default=0, 
                        help='which CUDA device to use (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    args = parser.parse_args()

    if args.artificial_missing_perc > 0:
        args.no_predictive = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_file = 'vi_{}_{}_{}person_{}item_{}maskperc_{}ability'.format(
        args.irt_model,
        args.dataset,
        args.num_person, 
        args.num_item,
        args.artificial_missing_perc,
        args.ability_dim, 
    )
    args.out_dir = os.path.join(args.out_dir, out_file) 
    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda: torch.cuda.set_device(args.gpu_device)

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True,
    )
    N_mini_batches = len(train_loader)
    if args.max_iters != -1:
        args.epochs = int(math.ceil(args.max_iters / float(len(train_loader))))
        print(f'Found MAX_ITERS={args.max_iters}, setting EPOCHS={args.epochs}')

    if args.irt_model == '1pl':
        model_class = VI_1PL
    elif args.irt_model == '2pl':
        model_class = VI_2PL
    elif args.irt_model == '3pl':
        model_class = VI_3PL
    else:
        raise Exception(f'model {args.irt_model} not recognized')

    model = model_class(
        args.ability_dim, 
        num_person,
        num_item,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def get_annealing_factor(epoch, which_mini_batch):
        if args.anneal_kl:
            annealing_factor = \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                 float(args.epochs // 2 * N_mini_batches))
        else:
            annealing_factor = args.beta_kl 
        return annealing_factor

    def train(epoch):
        model.train()
        train_loss = AverageMeter()
        pbar = tqdm(total=len(train_loader))

        for batch_idx, (index, response, _, mask) in enumerate(train_loader):
            mb = response.size(0)
            index = index.to(device)
            response = response.to(device)
            mask = mask.long().to(device)
            annealing_factor = get_annealing_factor(epoch, batch_idx)
        
            optimizer.zero_grad()
            outputs = model(index, response, mask)
            loss = model.elbo(*outputs, annealing_factor=annealing_factor)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), mb)

            pbar.update()
            pbar.set_postfix({'Loss': train_loss.avg})

        pbar.close()
        print('====> Train Epoch: {} Loss: {:.4f}'.format(epoch, train_loss.avg))

        return train_loss.avg

    def get_log_marginal_density(loader):
        model.eval()
        meter = AverageMeter()
        pbar = tqdm(total=len(loader))

        with torch.no_grad():
            for index, response, _, mask in loader:
                mb = response.size(0)
                index = index.to(device)
                response = response.to(device)
                mask = mask.long().to(device)

                marginal = model.log_marginal(
                    index,
                    response,
                    mask,
                    num_samples = args.num_posterior_samples,
                )

                marginal = torch.mean(marginal)
                meter.update(marginal.item(), mb)

                pbar.update()
                pbar.set_postfix({'Marginal': meter.avg})
        
        pbar.close()
        print('====> Marginal: {:.4f}'.format(meter.avg))

        return meter.avg

    def sample_posterior_predictive(loader):
        model.eval()
        meter = AverageMeter()
        pbar = tqdm(total=len(loader))

        with torch.no_grad():

            response_sample_set = []

            for index, response, _, mask in loader:
                mb = response.size(0)
                index = index.to(device)
                response = response.to(device)
                mask = mask.long().to(device)

                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                    model.encode(index, response, mask)
                
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

    def get_infer_dict(loader):
        model.eval()
        infer_dict = {}

        with torch.no_grad(): 
            ability_mus, item_feat_mus = [], []
            ability_logvars, item_feat_logvars = [], []

            # loop through and do inference for ability
            pbar = tqdm(total=len(loader))
            for index, response, _, mask in loader:
                mb = response.size(0)
                index = index.to(device)
                response = response.to(device)
                mask = mask.long().to(device)

                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                    model.encode(index, response, mask)

                ability_mus.append(ability_mu.cpu())
                ability_logvars.append(ability_logvar.cpu())

                item_feat_mus.append(item_feat_mu.cpu())
                item_feat_logvars.append(item_feat_logvar.cpu())

                pbar.update()

            ability_mus = torch.cat(ability_mus, dim=0)
            ability_logvars = torch.cat(ability_logvars, dim=0)

            item_feat_mus = torch.stack(item_feat_mus)
            item_feat_logvars = torch.stack(item_feat_logvars)

            pbar.close()

        infer_dict['ability_mu'] = ability_mus
        infer_dict['ability_logvar'] = ability_logvars
        infer_dict['item_feat_mu'] = item_feat_mus
        infer_dict['item_feat_logvar'] = item_feat_logvars

        return infer_dict

    is_best, best_loss = False, np.inf
    train_losses = np.zeros(args.epochs)
    train_times  = np.zeros(args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(epoch)
        end_time = time.time()
        train_losses[epoch] = train_loss
        train_times[epoch] = start_time - end_time
        
        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)

        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'args': args,
        }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'train_losses.npy'), train_losses)
        np.save(os.path.join(args.out_dir, 'train_times.npy'), train_times)

    for checkpoint_name in ['checkpoint.pth.tar', 'model_best.pth.tar']:
        checkpoint = torch.load(os.path.join(args.out_dir, checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])

        # redefine train loader w/o shuffle
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = args.batch_size, 
            shuffle = False,
        )

        infer_dict = get_infer_dict(train_loader)

        if not args.no_predictive:
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
                print(f'Missing Imputation Accuracy from samples: {missing_imputation_accuracy}')

        if not args.no_marginal:
            train_logp = get_log_marginal_density(train_loader)
            checkpoint['train_logp'] = train_logp

        checkpoint['infer_dict'] = infer_dict
        torch.save(checkpoint, os.path.join(args.out_dir, checkpoint_name))

    print(f'Saved to {args.out_dir}')
