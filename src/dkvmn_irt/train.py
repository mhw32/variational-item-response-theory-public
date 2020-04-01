import os
import time
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.distributions as dist
import torch.nn.functional as F

from src.dkvmn_irt.models import DKVMN_IRT
from src.datasets import load_dataset, artificially_mask_dataset
from src.utils import AverageMeter, save_checkpoint
from src.config import OUT_DIR, IS_REAL_WORLD


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='2pl_simulation',
                        choices=[
                            '1pl_simulation', 
                            '2pl_simulation', 
                            '3pl_simulation',
                            'critlangacq',
                            'duolingo',
                            'wordbank',
                            'pisa2015_science',
                        ],
                        help='which dataset to run on (default: 2pl_simulation)')
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
    parser.add_argument('--hidden-dim', type=int, default=50,
                        help='number of hidden dims (default: 50)')
    parser.add_argument('--max-num-person', help='limit the number of persons in dataset')
    parser.add_argument('--max-num-item', help='limit the number of items in dataset')

    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help='where to save chkpts (default: OUT_DIR)')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='default learning rate: 0.003')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--num-workers', type=int, default=0, 
                        help='number of workers for data loading (default: 0)')
    parser.add_argument('--max-grad-norm', type=float, default=10.0,
                        help='max of gradient norm (default: 10)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--gpu-device', type=int, default=0, 
                        help='which CUDA device to use (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    args = parser.parse_args()

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

    out_file = 'dkvmn_irt_pytorch_{}_{}person_{}item_{}maxperson_{}maxitem_{}ability_seed{}'.format(
        args.dataset,
        args.num_person, 
        args.num_item,
        args.max_num_person,
        args.max_num_item,
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
    test_dataset  = load_dataset(
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

    model = DKVMN_IRT(
        device,
        args.batch_size,
        num_item,
        args.hidden_dim,
        args.hidden_dim,
        args.hidden_dim,
        args.hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train(epoch):
        model.train()
        train_loss = AverageMeter()
        pbar = tqdm(total=len(train_loader))

        for batch_idx, (_, response, _, _) in enumerate(train_loader):
            mb = response.size(0)
            item_index = torch.arange(num_item).to(device)
            response = response.to(device)

            if mb != args.batch_size:
                pbar.update()
                continue

            with torch.no_grad():
                item_index = item_index.unsqueeze(0).repeat(mb, 1)
                item_index[(response == -1).squeeze(2)] = -1

                # build what dkvmn_irt expects
                q_data = item_index.clone()
                a_data = response.clone().squeeze(2)
                # ??? https://github.com/ckyeungac/DeepIRT/blob/master/load_data.py
                qa_data = q_data + a_data * num_item
                qa_data[(response == -1).squeeze(2)] = -1

                # map q_data and qa_data to 0 to N+1
                q_data = q_data + 1
                qa_data = qa_data + 1
                label = response.clone().squeeze(2)
       
            optimizer.zero_grad()
            pred_zs, student_abilities, question_difficulties = \
                model(q_data, qa_data, label)
            loss = model.get_loss(
                pred_zs,
                student_abilities,
                question_difficulties,
                label,
            )
            loss.backward()
            # https://github.com/ckyeungac/DeepIRT/blob/master/configs.py
            nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
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
            for _, response, _, _ in test_loader:
                mb = response.size(0)
                item_index = torch.arange(num_item).to(device)
                response = response.to(device)

                if mb != args.batch_size:
                    pbar.update()
                    continue

                with torch.no_grad():
                    item_index = item_index.unsqueeze(0).repeat(mb, 1)
                    item_index[(response == -1).squeeze(2)] = -1

                    # build what dkvmn_irt expects
                    q_data = item_index.clone()
                    a_data = response.clone().squeeze(2)
                    # ??? https://github.com/ckyeungac/DeepIRT/blob/master/load_data.py
                    qa_data = q_data + a_data * num_item
                    qa_data[(response == -1).squeeze(2)] = -1

                    # map q_data and qa_data to 0 to N+1
                    q_data = q_data + 1
                    qa_data = qa_data + 1
                    label = response.clone().squeeze(2) 

                pred_zs, student_abilities, question_difficulties = \
                    model(q_data, qa_data, label)
                loss = model.get_loss(
                    pred_zs,
                    student_abilities,
                    question_difficulties,
                    label,
                )
                test_loss.update(loss.item(), mb)

                pbar.update()
                pbar.set_postfix({'Loss': test_loss.avg})

        pbar.close()
        print('====> Test Epoch: {} Loss: {:.4f}'.format(epoch, test_loss.avg))

        return test_loss.avg

    is_best, best_loss = False, np.inf
    train_losses = np.zeros(args.epochs)
    test_losses  = np.zeros(args.epochs)
    train_times = np.zeros(args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(epoch)
        end_time = time.time()
        train_losses[epoch] = train_loss
        train_times[epoch] = start_time - end_time
        
        test_loss = test(epoch)
        test_losses[epoch] = test_loss
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'args': args,
        }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'train_losses.npy'), train_losses)
        np.save(os.path.join(args.out_dir, 'train_times.npy'), train_times)
        np.save(os.path.join(args.out_dir, 'test_losses.npy'),  test_losses)
