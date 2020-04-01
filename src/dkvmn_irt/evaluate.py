"""
Load and evaluate trained DKVMN_IRT model.
"""

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
    parser.add_argument('checkpoint_path', type=str, help='where to find trained checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)
    args = checkpoint['args']

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
        shuffle = False,
        num_workers = args.num_workers,
    )

    model = DKVMN_IRT(
        device,
        args.batch_size,
        num_item,
        args.hidden_dim,
        args.hidden_dim,
        args.hidden_dim,
        args.hidden_dim,
    ).to(device)

    # load trained parameters
    model.load_state_dict(checkpoint['model_state_dict'])

    def collect_predictions():
        model.eval()
        pbar = tqdm(total=len(train_loader))

        preds = []

        with torch.no_grad():
            for _, response, _, _ in train_loader:
                mb = response.size(0)
                item_index = torch.arange(num_item).to(device)
                response = response.to(device)

                if mb != args.batch_size:
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

                pred_logits, student_ability, question_difficulty = \
                    model(q_data, qa_data, label)

                preds_ = torch.sigmoid(pred_logits)
                preds.append(preds_)

                pbar.update()

        pbar.close()

        preds = torch.cat(preds, dim=0).detach().cpu()
        return preds

    if args.artificial_missing_perc > 0:
        missing_indices = train_dataset.missing_indices
        missing_labels = train_dataset.missing_labels

        if np.ndim(missing_labels):
            missing_labels = missing_labels[:, np.newaxis]

        inferred_response = collect_predictions()
        inferred_response = torch.round(inferred_response).long()

        correct, count = 0, 0
        true_labels, pred_labels = [], []
        for missing_index, missing_label in zip(missing_indices, missing_labels):
            if missing_index[0] < inferred_response.shape[0]:
                inferred_label = inferred_response[missing_index[0], missing_index[1]]
                if inferred_label.item() == missing_label[0]:
                    correct += 1
                count += 1
                pred_labels.append(inferred_label)
                true_labels.append(missing_label)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        missing_imputation_accuracy = correct / float(count)
        print(missing_imputation_accuracy)
    else:
        print('No missing data. Nothing to do.')
