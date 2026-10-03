import os
import numpy as np
import json
from dataset import GTZANDataset, get_dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from hear21passt.base30sec import load_model

from dattri.algorithm.tracin import TracInAttributor
from dattri.task import AttributionTask

import argparse
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TracIn calculation for a given stratify fold ID"
    )
    parser.add_argument(
        "--stratified_id",
        type=int,
        required=True,
        help="Stratify fold ID"
    )
    return parser.parse_args()


def main():

    args = parse_args()
    stratified_id = args.stratified_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model from checkpoint
    checkpoint_file = f"./stratified_models/fold_{stratified_id}/checkpoint_epoch_10_seed_42.pt"
    checkpoint = torch.load(checkpoint_file, map_location = torch.device(device))

    batch_size = 32
    dropout = 0.0001
    lr = 1e-4
    random_seed=42

    model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10) #10 total classes
        ).to(device)

    model = model.to(device)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k 
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()


    # testing loop
    embed_model = load_model(mode="embed_only").to(device)
    embed_model.eval()

    checkpoint_paths = [f"./stratified_models/fold_{stratified_id}/checkpoint_epoch_{n}_seed_42.pt" for n in range(11)]
    data_prefix = f"./stratified_splits_shuffled/fold_{stratified_id}/"

    # load training data
    train_loader = get_dataloader(data_prefix+'train.csv', embed_model, 1, shuffle=False, keep_wav=False, num_workers=0, use_embed=True)

    models = []
    models_state_dicts = []

    for path in checkpoint_paths:
        checkpoint_model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10) #10 total classes
        ).to(device)

        ckpt = torch.load(path, map_location = torch.device(device))
        state_dict = ckpt['model_state_dict']
        new_state_dict = {}

        for k, v in state_dict.items():
            # remove leading 'model.' if present
            new_key = k.replace("model.", "") if k.startswith("model.") else k
            new_state_dict[new_key] = v

        checkpoint_model.load_state_dict(new_state_dict)
        checkpoint_model.eval()                        # set to eval mode
        models.append(checkpoint_model)
        models_state_dicts.append(new_state_dict)

    # Running TracIn attributors
    normalized_grad = False
    ensemble = 11 # number of total checkpoints

    # loss function (referenced from dattri's benchmark_result.py)
    def loss_tracin(params, data_target_pair):
        data, label = data_target_pair
        data_t = data
        label_t = label.unsqueeze(0)
        loss = torch.nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, data_t)
        return loss(yhat, label_t.long())


    # attribution task
    task = AttributionTask(model=model.to(device),
                        loss_func=loss_tracin,
                        checkpoints=models_state_dicts)
    attributor = TracInAttributor(task=task,
                                weight_list=torch.ones(ensemble)*1e-3,
                                normalized_grad=False,
                                device=device)

    # Save TracIn results 
    all_self_scores = []

    for data, label in train_loader:
        single_dataset = TensorDataset(data, label)
        single_loader = DataLoader(single_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            score = attributor.attribute(single_loader, single_loader)
        
        all_self_scores.append(score.item())

    all_self_scores = torch.tensor(all_self_scores).numpy()

    save_path = f"self_attri/tracin/fold_{stratified_id}.npy"
    np.save(save_path, all_self_scores)
    print(f"Saved TracIn results for fold {stratified_id}")

if __name__ == "__main__":
    main()