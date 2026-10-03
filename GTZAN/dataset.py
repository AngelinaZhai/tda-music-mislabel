import torch
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import pandas as pd
from hear21passt.base30sec import load_model, get_scene_embeddings


DATA_ROOT = "./gtzan"

class GTZANDataset(Dataset):
    def __init__(self, csv_path, model, sr=22050, half=False, use_embed=False):
        self.fl = pd.read_csv(csv_path)
        self.model = model
        self.sr = sr
        self.half = half #if half model, randomly init only half of the training
        self.use_embed = use_embed

    def __len__(self):
        return len(self.fl)

    def __getitem__(self, idx):
        row = self.fl.iloc[idx]
        path = row["filepath"]
        label = row["label"]
        filename = row["filename"]
        target_samples = 32000 * 30 

        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        audio = librosa.resample(audio, orig_sr=self.sr, target_sr=32000)
        audio = torch.tensor(audio).float()

        length = audio.shape[-1]
        if length < target_samples:
            pad_len = target_samples - length
            audio = torch.nn.functional.pad(audio, (0, pad_len))
        else:
            audio = audio[:target_samples]

        if self.use_embed:
            emb = get_scene_embeddings(audio.unsqueeze(0).to('cpu'), self.model)
            return emb.to('cpu'), torch.tensor(label), filename
        else:
            return audio.to('cpu'), torch.tensor(label), filename

    @staticmethod
    def collate(batch):
        audios = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return audios, labels

    @staticmethod
    def collate_with_filename(batch):
        audios = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        filenames = [item[2] for item in batch]
        return audios, labels, filenames


def get_dataloader(csv_path, model, batch_size=32, shuffle=True,
                    num_workers=1, keep_wav=False, half=False, use_embed=False,):

    dataset = GTZANDataset(csv_path, model, use_embed=use_embed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=GTZANDataset.collate_with_filename
        if keep_wav else GTZANDataset.collate
    )
