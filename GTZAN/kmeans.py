from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import mode


# metadata_file = './purified_train.parquet'

n_classes = 10

for stratify_id in range(0, 10):

    # load data of fold
    data_np = np.load(f'compiled_data/fold_{stratify_id}/data.npy')
    data_np = data_np.squeeze(axis=1)

    data_labels = np.load(f'compiled_data/fold_{stratify_id}/labels.npy')
    all_names = np.load(f'compiled_data/fold_{stratify_id}/names.npy')

    scaler = StandardScaler()
    data = scaler.fit_transform(data_np)

    classes = np.unique(data_labels)
    centroids = np.array([data[data_labels == c].mean(axis=0) for c in classes])

    kmeans = KMeans(n_clusters=len(classes), init=centroids, n_init=1)
    kmeans.fit(data)

    # link clusters to original labels
    cluster_labels = kmeans.labels_ # get predicted labels
    cm = confusion_matrix(data_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize overlap
    mapping = dict(zip(col_ind, row_ind))
    aligned = np.array([mapping[c] for c in cluster_labels])
    accuracy = (aligned == data_labels).mean()
    print(f"Accuracy: {accuracy:.4f}")

    # distance from each datapoint to its assigned centroid and all centroids
    final_centroids = kmeans.cluster_centers_ # (n_clusters, n_features)
    distances_to_all = np.linalg.norm(
        data[:, np.newaxis, :] - final_centroids[np.newaxis, :, :], # (N, n_clusters, n_features)
        axis=2
    )# (N, n_clusters)
    distances_to_assigned = distances_to_all[np.arange(len(data)), cluster_labels] # (N,)

    # Distance from each point to its true class centroid (not assigned)
    dist_to_true_centroid = distances_to_all[np.arange(len(data)), data_labels]

    # Majority voting: for each cluster, find the majority label
    # flag points whose true label disagrees with their cluster's majority
    cluster_majority = {}
    for c in np.unique(cluster_labels):
        labels_in_cluster = data_labels[cluster_labels == c]
        majority = mode(labels_in_cluster, keepdims=True).mode[0]
        cluster_majority[c] = majority

    majority_label = np.array([cluster_majority[c] for c in cluster_labels])  # (N,)
    majority_vote_anomaly = (data_labels != majority_label).astype(int)        # 1 = anomaly

    results_df = pd.DataFrame({
        'filename': np.array(all_names),
        'true_label': data_labels,
        'assigned_cluster': cluster_labels,
        'assigned_label': aligned,
        'dist_to_assigned_centroid': distances_to_assigned,
        'dist_to_true_centroid': dist_to_true_centroid,
        'cluster_majority_label': majority_label,
        'majority_vote_anomaly': majority_vote_anomaly,  # 1 = flagged as mislabeled
    })

    results_df.to_csv(f'kmeans_mislabel_fold_{stratify_id}.csv', index=False)
