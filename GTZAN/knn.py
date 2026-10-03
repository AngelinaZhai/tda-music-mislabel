import numpy as np
import os

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
    
k = 5

for stratify_id in range(0, 10):

    data_np = np.load(f'compiled_data/fold_{stratify_id}/data.npy')
    data_np = data_np.squeeze(axis=1)

    data_labels = np.load(f'compiled_data/fold_{stratify_id}/labels.npy')
    all_names = np.load(f'compiled_data/fold_{stratify_id}/names.npy')

    scaler = StandardScaler()
    data_normed = scaler.fit_transform(data_np)

    nbrs = NearestNeighbors(n_neighbors=k).fit(data_normed)
    distances, indices = nbrs.kneighbors(data_normed)

    avg_distance = distances[:, 1:].mean(axis=1)

    results_df = pd.DataFrame({
        'filename': all_names,
        'label': data_labels,
        'avg_neighbor_dist': avg_distance,
    })

    results_df.to_csv(f"ml_results/knn_raw/k_{k}/fold_{stratify_id}.csv", index=False)