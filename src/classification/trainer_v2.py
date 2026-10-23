"""
Controlled Food Classification Training Suite (V2)
Follows official PyTorch Transfer Learning Tutorial:
https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Implements:
  - Candidate C: Head-Only Transfer Learning (Backbone Frozen, Training 5-Class Head)
  - Candidate D: Staged Fine-Tuning (Unfreezing Upper Bottleneck Blocks with Lower LR & Cosine Decay)

Strictly uses validation split for checkpoint selection and early stopping.
The test split is held-out and completely untouched during training and tuning.
"""

import os
import sys
import json
import time
import random
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

FOOD_CLASSES = [
    "apple_fresh_raw",
    "banana_raw",
    "broccoli_steamed",
    "chicken_breast_grilled",
    "rice_white_cooked"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(FOOD_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(FOOD_CLASSES)}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FoodDataset(Dataset):
    def __init__(self, items: list, base_dir: str, transform=None):
        self.items = items
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.base_dir, item["file_path"])
        img = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[item["canonical_id"]]
        if self.transform:
            img = self.transform(img)
        return img, label, item["canonical_id"], item["image_id"]

def create_dataloaders(base_dir: str, batch_size: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    manifest_path = os.path.join(base_dir, "data_manifests", "food_dataset_v2_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run expand_dataset_v2.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_items = [r for r in data["images"] if r["split"] == "train"]
    val_items = [r for r in data["images"] if r["split"] == "val"]
    test_items = [r for r in data["images"] if r["split"] == "test"]

    # ImageNet standard normalization
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Data augmentation for training: rotation, scaling, crop, color jitter
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Deterministic evaluation transform for validation and testing
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_ds = FoodDataset(train_items, base_dir, train_transform)
    val_ds = FoodDataset(val_items, base_dir, eval_transform)
    test_ds = FoodDataset(test_items, base_dir, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    split_info = {
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "train_items": train_items,
        "val_items": val_items,
        "test_items": test_items
    }
    return train_loader, val_loader, test_loader, split_info

def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels, _, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def train_candidate_c_head_only(base_dir: str, epochs: int = 15, lr: float = 1e-3, batch_size: int = 8) -> Dict[str, Any]:
    """
    Candidate C: Head-Only Transfer Learning on Expanded Dataset V2.
    Frozen MobileNetV3-Small features backbone; train classifier head.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 75)
    print("TRAINING CANDIDATE C: HEAD-ONLY TRANSFER LEARNING (DATASET V2)")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr} | Batch Size: {batch_size}")
    print("=" * 75)

    train_loader, val_loader, _, split_info = create_dataloaders(base_dir, batch_size=batch_size)
    print(f"Dataset Splits: Train={split_info['train_count']}, Val={split_info['val_count']}, Test={split_info['test_count']}")

    # Load official pretrained MobileNetV3-Small
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze all feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier output layer with 5 food classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(FOOD_CLASSES))
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters (Head Only): {trainable_params:,} ({trainable_params / (trainable_params + frozen_params):.2%})")
    print(f"Frozen Parameters (Backbone):     {frozen_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)

    history = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    ckpt_dir = os.path.join(base_dir, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "mobilenet_v3_food_head_only_v2.pth")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for imgs, labels, _, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / total_train
        train_acc = train_correct / total_train

        val_loss, val_acc = evaluate_split(model, val_loader, device)
        history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4)
        })

        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_mapping": CLASS_TO_IDX,
                "candidate": "Candidate C (Head-Only V2)",
                "dataset_version": "2.0.0"
            }, best_ckpt_path)
            saved_tag = "[BEST CHECKPOINT SAVED]"
        else:
            saved_tag = ""

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} {saved_tag}")

    elapsed = time.time() - t_start
    print(f"\nCandidate C Complete in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc:.2%} (Loss: {best_val_loss:.4f})")
    return {
        "candidate": "Candidate C (Head-Only V2)",
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 2),
        "best_val_acc": best_val_acc,
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint_path": "models/checkpoints/mobilenet_v3_food_head_only_v2.pth",
        "history": history
    }

def train_candidate_d_staged_finetuning(base_dir: str, epochs: int = 12, lr: float = 1e-4, batch_size: int = 8) -> Dict[str, Any]:
    """
    Candidate D: Staged Fine-Tuning of Selected Upper Bottleneck Layers.
    Initializes from Candidate C best checkpoint.
    Unfreezes features[11], features[12] and classifier with lower LR and cosine annealing.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 75)
    print("TRAINING CANDIDATE D: STAGED FINE-TUNING OF UPPER LAYERS (DATASET V2)")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr} | Batch Size: {batch_size}")
    print("=" * 75)

    train_loader, val_loader, _, split_info = create_dataloaders(base_dir, batch_size=batch_size)

    # Initialize model with 5 classes
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(FOOD_CLASSES))

    # Initialize from Candidate C best checkpoint
    head_ckpt = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_head_only_v2.pth")
    if not os.path.exists(head_ckpt):
        raise FileNotFoundError(f"Candidate C checkpoint missing: {head_ckpt}. Run Candidate C first.")
    ckpt_data = torch.load(head_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])

    # Freeze lower feature extractor (layers 0 through 10)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Unfreeze upper two bottleneck blocks (layers 11 and 12)
    for param in model.features[11].parameters():
        param.requires_grad = True
    for param in model.features[12].parameters():
        param.requires_grad = True
    
    # Full classifier head is unfrozen
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters (Upper Bottleneck + Head): {trainable_params:,} ({trainable_params / (trainable_params + frozen_params):.2%})")
    print(f"Frozen Parameters (Lower Backbone):           {frozen_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_finetuned_v2.pth")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for imgs, labels, _, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        avg_train_loss = train_loss / total_train
        train_acc = train_correct / total_train

        val_loss, val_acc = evaluate_split(model, val_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(cur_lr, 7)
        })

        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_mapping": CLASS_TO_IDX,
                "candidate": "Candidate D (Staged Fine-Tuned V2)",
                "dataset_version": "2.0.0"
            }, best_ckpt_path)
            saved_tag = "[BEST CHECKPOINT SAVED]"
        else:
            saved_tag = ""

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | LR: {cur_lr:.2e} {saved_tag}")

    elapsed = time.time() - t_start
    print(f"\nCandidate D Complete in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc:.2%} (Loss: {best_val_loss:.4f})")
    return {
        "candidate": "Candidate D (Staged Fine-Tuned V2)",
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 2),
        "best_val_acc": best_val_acc,
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint_path": "models/checkpoints/mobilenet_v3_food_finetuned_v2.pth",
        "history": history
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res_c = train_candidate_c_head_only(base, epochs=15)
    res_d = train_candidate_d_staged_finetuning(base, epochs=12)
    print("\nTraining Suite V2 complete.")
