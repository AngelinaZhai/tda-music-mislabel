from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from hear21passt.base30sec import load_model

from dataset import GTZANDataset, get_dataloader

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AUM calculation for a given stratify fold ID"
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

    # purified_portion = [5, 10, 20]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=64
    dropout=0

    aum_accumulator = defaultdict(list)  

    # load all checkpoints of this fold
    model_list = [f"./stratified_models/fold_{stratified_id}/checkpoint_epoch_{n}_seed_42.pt" for n in range(11)]

    embed_model = load_model(mode="embed_only").to(device)
    train_loader = get_dataloader(f"stratified_splits_shuffled/fold_{stratified_id}/train.csv", embed_model, batch_size, shuffle=False, num_workers = 0, keep_wav=True, use_embed=True)

    for model_path in model_list:
        model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        ).to(device)

        checkpoint = torch.load(model_path, map_location=torch.device(device))
        new_state_dict = {
            (k.replace("model.", "") if k.startswith("model.") else k): v
            for k, v in checkpoint['model_state_dict'].items()
        }
        model.load_state_dict(new_state_dict)
        model.eval()

        with torch.no_grad():
            
            for batch in train_loader:
                emb, labels, fnames = batch
                emb = emb.to(device)
                labels = labels.to(device)
                logits = model(emb)

                logits = logits.squeeze(1)

                true_logits = logits[torch.arange(len(labels)), labels]  # (B,)
                masked_logits = logits.clone()
                masked_logits[torch.arange(len(labels)), labels] = float('-inf')
                max_other_logits = masked_logits.max(dim=1).values       # (B,)
                margins = (true_logits - max_other_logits).cpu().numpy() # (B,)

                # One margin per sample per checkpoint, append directly
                for fname, margin in zip(fnames, margins):
                    aum_accumulator[fname].append(float(margin))

    n_checkpoints = len(model_list)
    aum_records = []
    for fname, margins in aum_accumulator.items():
        assert len(margins) == n_checkpoints, f"{fname} has {len(margins)} margins, expected {n_checkpoints}"
        aum_records.append({"fname": fname, "aum": float(np.mean(margins))})

    aum_df = pd.DataFrame(aum_records)
    aum_df.to_csv(f"aum_results_{stratified_id}.csv", index=False)
    print(f"Saved AUM for fold {stratified_id}: {len(aum_df)} samples")

if __name__ == "__main__":
    main()