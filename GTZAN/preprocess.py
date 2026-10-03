'''
Preprocesses the GTZAN dataset to create a metadata CSV and stratified train/val/test splits.
'''

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


DATA_ROOT = "./gtzan"
all_data_path = "metadata_all.csv"


def make_metadata(root, csv_path=None):
    genres = sorted(os.listdir(root))
    label_map = {genre: idx for idx, genre in enumerate(genres)}
    data = []

    for genre in genres:
        folder = os.path.join(root, genre)
        for fname in os.listdir(folder):
            if fname.endswith(".wav"):
                full_path = os.path.join(folder, fname)
                data.append({
                    "filepath": full_path,
                    "label": label_map[genre],
                    "filename": fname
                })

    df = pd.DataFrame(data)

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata to {csv_path}")

    return df, label_map


def make_splits(
    csv_path,
    n_splits=10,
    output_dir="stratified_splits",
    random_state=42,
):
    """
    Replicates the split strategy from:
      Thanh Chu Ba et al. (ScienceDirect, 2025)
      "Music genre classification using deep neural networks and data augmentation"

    Strategy:
      - Outer loop : 10-fold stratified CV → 1 fold = test, 9 folds = train+val pool
      - Inner split : train+val pool split at a 1:8 (val:train) ratio using
                      a second StratifiedKFold with 9 inner folds, iterated
                      fully so every sample in the pool is val exactly once
      - Result      : across all 10 folds, each sample is:
                        test  in exactly 1 fold
                        val   in exactly 1 fold  (1 of its 9 pool appearances)
                        train in exactly 8 folds (8 of its 9 pool appearances)

    For GTZAN (1000 samples, 10 genres, 100 per genre):
      test  : 100 samples (10 per genre)
      val   : 100 samples (10 per genre)
      train : 800 samples (80 per genre)
    """
    df = pd.read_csv(csv_path).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)

    outer_skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    # track (original_df_index → role) per fold
    col_names = [f"fold_{i}" for i in range(n_splits)]
    summary = pd.DataFrame("—", index=df.index, columns=col_names)

    # pre-compute all outer splits so we can assign inner val folds
    # deterministically: outer fold i → inner val fold i (mod n_inner)
    outer_splits = list(outer_skf.split(df, df["label"]))

    folds = []

    for fold_idx, (pool_idx, test_idx) in enumerate(outer_splits):
        col = f"fold_{fold_idx}"

        test_df = df.iloc[test_idx].copy()
        summary.loc[test_df.index, col] = "test"

        pool_df = df.iloc[pool_idx].reset_index(drop=False)  # keep original index

        # inner 9-fold on the pool — always use the same random_state so
        # fold assignment is deterministic and covers the full pool
        n_inner = n_splits - 1   # 9
        inner_skf = StratifiedKFold(
            n_splits=n_inner, shuffle=True, random_state=random_state
        )
        inner_splits = list(inner_skf.split(pool_df, pool_df["label"]))

        # rotate which inner fold is val based on outer fold index
        # → each sample gets a different val assignment each time it's in the pool
        val_inner_fold = fold_idx % n_inner
        train_inner_idx, val_inner_idx = inner_splits[val_inner_fold]

        train_df = pool_df.iloc[train_inner_idx]
        val_df   = pool_df.iloc[val_inner_idx]

        # map back to original df indices for summary
        summary.loc[train_df["index"], col] = "train"
        summary.loc[val_df["index"],   col] = "val"

        # drop helper index column before saving
        train_out = train_df.drop(columns="index")
        val_out   = val_df.drop(columns="index")

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        train_out.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_out.to_csv(  os.path.join(fold_dir, "val.csv"),   index=False)
        test_df.to_csv(  os.path.join(fold_dir, "test.csv"),  index=False)

        folds.append({"train": train_out, "val": val_out, "test": test_df})

        print(
            f"Fold {fold_idx}: "
            f"train={len(train_out):>4}  "
            f"val={len(val_out):>4}  "
            f"test={len(test_df):>4}"
        )

    # ── summary ────────────────────────────────────────────────────────────
    role_cols = summary[col_names]
    summary = pd.concat([df[["filepath", "filename", "label"]], summary], axis=1)
    summary["n_train"] = (role_cols == "train").sum(axis=1)
    summary["n_val"]   = (role_cols == "val").sum(axis=1)
    summary["n_test"]  = (role_cols == "test").sum(axis=1)

    summary_path = os.path.join(output_dir, "fold_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")
    print("\nRole counts across all folds:")
    print(summary[["n_train", "n_val", "n_test"]].describe().loc[["min", "max"]])

    return folds, summary

if __name__ == "__main__":
    # make_metadata(DATA_ROOT, all_data_path) #commented out bc already compiled
    folds, summary = make_splits(all_data_path, output_dir="stratified_splits")