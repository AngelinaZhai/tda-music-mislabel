import torch
import numpy as np
from dataset import GTZANDataset
from hear21passt.base30sec import load_model
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = load_model(mode="embed_only").to(device)
embed_model.eval()


for stratified_id in range(0, 10):

    all_data, all_labels = [], []
    all_batch, all_idx = [], []
    all_names = [] #for actual file names

    data_prefix = f"./stratified_splits_shuffled/fold_{stratified_id}/"

    dataset = GTZANDataset(data_prefix+'train.csv', embed_model, use_embed=True)


    for idx in range(len(dataset)):
        data = dataset[idx][0] # embedding
        label = dataset[idx][1] # perturbed if in map, else original
        fname = dataset[idx][2]  # filename from fnames
        all_data.append(data.detach().cpu().numpy())
        all_labels.append(label)
        all_names.append(fname)
        all_idx.append(idx)


    data_np = np.stack(all_data, axis=0) # (n_samples, n_features)
    save_dir = f"./compiled_data/fold_{stratified_id}"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "data.npy"), data_np)
    np.save(os.path.join(save_dir, "labels.npy"), np.array(all_labels))
    np.save(os.path.join(save_dir, "indices.npy"), np.array(all_idx))
    np.save(os.path.join(save_dir, "names.npy"), np.array(all_names))

    print(f"Completed processing fold {stratified_id}")
