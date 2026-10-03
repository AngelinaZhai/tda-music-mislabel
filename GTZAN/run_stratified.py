import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from hear21passt.base30sec import load_model, get_scene_embeddings

import os
import random
import numpy as np

import argparse

from dataset import GTZANDataset, get_dataloader


batch_size = 32
dropout=0.3
lr = 5e-4
epochs = 11
random_seed=42


def set_seed(seed: int = 42):
    random.seed(seed)                        
    np.random.seed(seed)                     
    torch.manual_seed(seed)                  
    torch.cuda.manual_seed(seed)             
    torch.cuda.manual_seed_all(seed)         
    torch.backends.cudnn.deterministic = False #deterministic but slower
    torch.backends.cudnn.benchmark = True #false disables gpu autotuner (whic introduces randomness)



#training loop
def train_epoch(embed_model, classifier, train_loader, criterion, optimizer, device, epoch):
    classifier.train()
    total_loss = 0
    total_correct = 0
    total = 0

    for audio, label in train_loader:
        
        emb = get_scene_embeddings(audio, embed_model).to(device) 
        label = label.to(device)

        optimizer.zero_grad()
        output = classifier(emb)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        #calculate loss
        _, pred = output.max(1)
        total_loss += loss.item()*emb.size(0)
        total_correct += (pred == label).sum().item()
        total += label.size(0)
    
    # get avg loss per epoch
    avg_loss = total_loss/total
    accuracy = total_correct/total
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }

    os.makedirs(f'./stratified_models/fold_{stratify_id}', exist_ok=True)
    torch.save(checkpoint, f'./stratified_models/fold_{stratify_id}/checkpoint_epoch_{epoch}_seed_{random_seed}.pt')

    # print(f"Epoch {epoch} Train | Loss: {avg_loss}, Acc:{accuracy}")

    return avg_loss, accuracy


#validation loop
def val_epoch(embed_model, classifier, val_loader, criterion, optimizer, device):
    classifier.eval()
    total_loss = 0
    total_correct = 0
    total = 0

    for audio, label in val_loader:
        emb = get_scene_embeddings(audio, embed_model).to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = classifier(emb)
        loss = criterion(output, label)

        #calculate loss
        _, pred = output.max(1)
        total_loss += loss.item()*emb.size(0)
        total_correct += (pred == label).sum().item()
        total += label.size(0)
    
    #get avg loss per epoch
    avg_loss = total_loss/total
    accuracy = total_correct/total

    return avg_loss, accuracy


#training loop
def train_model(embed_model, classifier, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda"):
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    #lists for storing loss & acc
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        t_loss, t_acc = train_epoch(embed_model, classifier, train_loader, criterion, optimizer, device, epoch)
        v_loss, v_acc = val_epoch(embed_model, classifier, val_loader, criterion, optimizer, device)
        
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        val_loss.append(v_loss)
        val_acc.append(v_acc)

        print(f"Epoch {epoch} Train | Loss: {t_loss:.4f}, Acc: {t_acc:.4f}")
        print(f"Epoch {epoch} Validation | Loss: {v_loss:.4f}, Acc:{v_acc:.4f}")

    return classifier

#testing loop
def test_model(embed_model, classifier, test_loader, device="cuda"):
    classifier.eval()
    total_correct = 0
    total = 0

    with torch.no_grad():
        for audios, labels in test_loader:
            # embs = get_scene_embeddings(audios, embed_model).mean(dim=1)
            embs = get_scene_embeddings(audios, embed_model)
            embs, labels = embs.to(device), labels.to(device)

            outputs = classifier(embs)
            _, preds = outputs.max(1)

            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = total_correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save GTZAN classifier given stratified ID"
    )
    parser.add_argument(
        "--stratify_id",
        type=int,
        required=True,
        help="Stratify ID"
    )

    return parser.parse_args()

if __name__=="__main__":
    set_seed(random_seed)
    print(f"Set random seed: {random_seed}")

    args = parse_args()
    global stratify_id
    stratify_id = args.stratify_id
    print(f"Training model on stratify group {stratify_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #get passt embedding model 
    embed_model = load_model(mode="embed_only").to(device)
    embed_model.eval()

    # Models are trained on SHUFFLED version!
    train_loader = get_dataloader(f"stratified_splits_shuffled/fold_{stratify_id}/train.csv", embed_model, batch_size) 
    val_loader   = get_dataloader(f"stratified_splits_shuffled/fold_{stratify_id}/val.csv", embed_model, batch_size)
    test_loader = get_dataloader(f"stratified_splits_shuffled/fold_{stratify_id}/test.csv", embed_model, batch_size)

    classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10) #10 total classes
    ).to(device)

    trained_model = train_model(embed_model, classifier, train_loader, val_loader, epochs, lr, device=device)
    test_acc = test_model(embed_model, trained_model, test_loader, device=device)