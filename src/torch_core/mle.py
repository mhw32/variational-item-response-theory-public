"""
Train IRT with Maximum Likelihood
http://douglasrizzo.com.br/catsim/estimation.html#catsim.estimation.DifferentialEvolutionEstimator
"""

import os
import time
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.distributions as dist
import torch.nn.functional as F

from src.torch_core.models import MLE_1PL, MLE_2PL, MLE_3PL
from src.datasets import load_dataset, artificially_mask_dataset
from src.utils import AverageMeter, save_checkpoint
from src.config import OUT_DIR, IS_REAL_WORLD


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

    parser.add_argument('--no-infer-dict', action='store_true', default=False,
                        help='if true, skip infer dict collection (default: False)')
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='if true, skip test (default: False)')
    parser.add_argument('--num-person', type=int, default=1000,
                        help='number of people in data (default: 1000)')
    parser.add_argument('--num-item', type=int, default=100,
                        help='number of people in data (default: 100)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='number of hidden dims (default: 64)')
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
    parser.add_argument('--num-workers', type=int, default=0, 
                        help='number of workers for data loading (default: 0)')
    parser.add_argument('--max-iters', type=int, default=-1, metavar='N',
                        help='number of maximum iterations (default: -1)')
    parser.add_argument('--artificial-missing-perc', type=float, default=0.,
                        help='how much to blank out so we can measure acc (default: 0)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--gpu-device', type=int, default=0, 
                        help='which CUDA device to use (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    args = parser.parse_args()

    if args.artificial_missing_perc:
        args.no_infer_dict = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    out_file = 'mle_{}_{}_{}person_{}item_{}maxperson_{}maxitem_{}maskperc_{}ability_seed{}'.format(
        args.irt_model, 
        args.dataset,
        args.num_person, 
        args.num_item,
        args.max_num_person,
        args.max_num_item,
        args.artificial_missing_perc,
        args.ability_dim, 
        args.seed,
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
    test_dataset = load_dataset(
        args.dataset,
        train = False, 
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
        num_workers = args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = args.batch_size, 
        shuffle = False,
        num_workers = args.num_workers,
    )
    N_mini_batches = len(train_loader)
    if args.max_iters != -1:
        args.epochs = int(math.ceil(args.max_iters / float(len(train_loader))))
        print(f'Found MAX_ITERS={args.max_iters}, setting EPOCHS={args.epochs}') 

    if args.irt_model == '1pl':
        model_class = MLE_1PL
    elif args.irt_model == '2pl':
        model_class = MLE_2PL
    elif args.irt_model == '3pl':
        model_class = MLE_3PL
    else:
        raise Exception(f'model {args.irt_model} not recognized')

    model = model_class(
        args.ability_dim,
        num_person,
        num_item,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train(epoch):
        model.train()
        train_loss = AverageMeter()
        pbar = tqdm(total=len(train_loader))

        for batch_idx, (index, response, _, mask) in enumerate(train_loader):
            mb = response.size(0)
            index = index.to(device)
            response = response.to(device)
            mask = mask.long().to(device)
        
            optimizer.zero_grad()
            response_mu = model(index, response, mask)
            loss = F.binary_cross_entropy(response_mu, response.float(), reduction='none')
            loss = loss * mask
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), mb)

            pbar.update()
            pbar.set_postfix({'Loss': train_loss.avg})

        pbar.close()
        print('====> Train Epoch: {} Loss: {:.4f}'.format(epoch, train_loss.avg))

        return train_loss.avg

    def test(epoch):
        model.eval()
        test_loss = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        with torch.no_grad():
            for index, response, _, mask in test_loader:
                mb = response.size(0)
                index = index.to(device)
                response = response.to(device)
                mask = mask.long().to(device)

                response_mu = model(index, response, mask)
                loss = F.binary_cross_entropy(response_mu, response.float())
                test_loss.update(loss.item(), mb)

                pbar.update()
                pbar.set_postfix({'Loss': test_loss.avg})

        pbar.close()
        print('====> Test Epoch: {} Loss: {:.4f}'.format(epoch, test_loss.avg))

        return test_loss.avg

    def get_infer_dict(loader):
        model.eval()
        infer_dict = {}

        with torch.no_grad(): 
            abilitys, item_feats = [], []

            pbar = tqdm(total=len(loader))
            for index, response, _, mask in loader:
                mb = response.size(0)
                index = index.to(device)
                response = response.to(device)
                mask = mask.to(device)

                ability, item_feat = model.encode(index, response, mask)

                abilitys.append(ability.cpu())
                item_feats.append(item_feat.cpu())

                pbar.update()

            abilitys = torch.cat(abilitys, dim=0)
            pbar.close()

        infer_dict['ability'] = abilitys
        infer_dict['item_feat'] = item_feats

        return infer_dict

    is_best, best_loss = False, np.inf
    train_losses = np.zeros(args.epochs)
    if not args.no_test:
        test_losses  = np.zeros(args.epochs)
    train_times = np.zeros(args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(epoch)
        end_time = time.time()
        train_losses[epoch] = train_loss
        train_times[epoch] = start_time - end_time
        
        if not args.no_test:
            test_loss = test(epoch)
            test_losses[epoch] = test_loss
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
        else:
            is_best = train_loss < best_loss
            best_loss = min(train_loss, best_loss)

        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'args': args,
            'total_iters': len(train_loader) * args.epochs,
        }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'train_losses.npy'), train_losses)
        np.save(os.path.join(args.out_dir, 'train_times.npy'), train_times)

        if not args.no_test:
            np.save(os.path.join(args.out_dir, 'test_losses.npy'),  test_losses)

    for checkpoint_name in ['checkpoint.pth.tar', 'model_best.pth.tar']:
        checkpoint = torch.load(os.path.join(args.out_dir, checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = args.batch_size, 
            shuffle = False,
        )

        if not args.no_infer_dict:
            infer_dict = get_infer_dict(train_loader)
            checkpoint['infer_dict'] = infer_dict

            if args.artificial_missing_perc > 0:
                missing_indices = train_dataset.missing_indices
                missing_labels = train_dataset.missing_labels

                if np.ndim(missing_labels) == 1:
                    missing_labels = missing_labels[:, np.newaxis]

                ability = infer_dict['ability']
                item_feat = infer_dict['item_feat']

                inferred_response = model.decode(ability, item_feat[0])
                inferred_response = torch.round(inferred_response)

                correct, count = 0, 0
                for missing_index, missing_label in zip(missing_indices, missing_labels):
                    inferred_label = inferred_response[missing_index[0], missing_index[1]]
                    if inferred_label.item() == missing_label[0]:
                        correct += 1
                    count += 1

                missing_imputation_accuracy = correct / float(count)
                checkpoint['missing_imputation_accuracy'] = missing_imputation_accuracy

        torch.save(checkpoint, os.path.join(args.out_dir, checkpoint_name))
