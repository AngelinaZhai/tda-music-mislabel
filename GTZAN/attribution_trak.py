import os
import numpy as np
from dataset import GTZANDataset, get_dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from hear21passt.base30sec import load_model

import argparse
import warnings
warnings.filterwarnings('ignore')

from dattri.algorithm.trak import TRAKAttributor
from dattri.task import AttributionTask



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TRAK calculation for a given stratify fold ID"
    )
    parser.add_argument(
        "--stratified_id",
        type=int,
        required=True,
        help="Stratify fold ID"
    )

    return parser.parse_args()


def main():
    device = 'cpu'
    
    args = parse_args()
    stratified_id = args.stratified_id

    # load model from checkpoint
    checkpoint_file = f"./stratified_models/fold_{stratified_id}/checkpoint_epoch_10_seed_42.pt"
    checkpoint = torch.load(checkpoint_file, map_location = torch.device(device))

    batch_size = 32
    dropout = 0.0001
    lr = 1e-4
    epochs = 25
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

    # load training data
    checkpoint_paths = [f"./stratified_models/fold_{stratified_id}/checkpoint_epoch_{n}_seed_42.pt" for n in range(11)]
    data_prefix = f"./stratified_splits_shuffled/fold_{stratified_id}/"

    train_loader = get_dataloader(data_prefix+'train.csv', embed_model, 4, shuffle=False, keep_wav=False, num_workers=0, use_embed=True)


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
        checkpoint_model.eval()        
        models.append(checkpoint_model)
        models_state_dicts.append(new_state_dict)

    # loss function (referenced from dattri's benchmark_result.py)
    def loss_trak(params, data_target_pair):
        data, label = data_target_pair
        data_t = data
        label_t = label.unsqueeze(0)
        loss = torch.nn.CrossEntropyLoss()
        output = torch.func.functional_call(model, params, data_t)
        logp = -loss(output, label_t)
        return logp - torch.log(1 - torch.exp(logp))

    def m_trak(params, data_target_pair):
        data, label = data_target_pair
        data_t = data
        label_t = label.unsqueeze(0)
        loss = torch.nn.CrossEntropyLoss()
        output = torch.func.functional_call(model, params, data_t)
        return torch.exp(-loss(output, label_t.long()))

    projector_kwargs = {
        "proj_dim": 512,
        "proj_max_batch_size": 32,
        "proj_seed": 42,
        "device": 'cpu',
    }

    all_self_scores = []


    # attribution task
    task = AttributionTask(model=model.to(device),
                        loss_func=loss_trak,
                        checkpoints=models_state_dicts)

    attributor = TRAKAttributor(task=task,
                                correct_probability_func = m_trak,
                                device='cpu',
                                projector_kwargs=projector_kwargs,)

    with torch.no_grad():
        # Process each batch on its own
        for batch_idx, (data, label) in enumerate(train_loader):
            # Create single batch loader for attribution
            single_batch_loader = DataLoader(
                TensorDataset(data, label), 
                batch_size=batch_size,  # Use full batch
                shuffle=False
            )
            
            try:
                # Get attribution for this batch
                score = attributor.attribute(single_batch_loader, single_batch_loader)
                
                # Extract diagonal (self-attribution for each sample)
                batch_scores = score.diagonal().cpu().numpy()
                all_self_scores.extend(batch_scores)
                
                print(f"Batch {batch_idx}: processed {len(batch_scores)} samples")
                
            except torch.linalg.LinAlgError as e:
                print(f"Batch {batch_idx} failed: {e}")
                continue

    # Save TRAK results
    all_self_scores = np.array(all_self_scores)

    save_path = f"self_attri/trak/fold_{stratified_id}.npy"

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    np.save(save_path, all_self_scores)


    print(f"Saved {len(all_self_scores)} TRAK scores, stratify_id = {stratified_id}")

if __name__ == "__main__":
    main()