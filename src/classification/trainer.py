"""
Controlled Fine-Tuning Pipeline for MobileNetV3-Small on 5 Food Classes
Implements Experiment 1 (Head-Only training) and Experiment 2 (Staged fine-tuning with upper layers unfrozen).
Saves checkpoints, training history, and loss metrics.
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

class FoodDataset(Dataset):
    def __init__(self, manifest_items: list, base_dir: str, transform=None):
        self.items = manifest_items
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        info = self.items[idx]
        img_path = os.path.join(self.base_dir, info["file_path"])
        img = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[info["canonical_id"]]
        if self.transform:
            img = self.transform(img)
        return img, label, info["canonical_id"]

def create_dataloaders(base_dir: str, batch_size: int = 8) -> Tuple[DataLoader, DataLoader, list, list]:
    manifest_path = os.path.join(base_dir, "data_manifests", "finetune_dataset_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_items = [r for r in data["images"] if r["split"] == "train"]
    val_items = [r for r in data["images"] if r["split"] == "val"]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = FoodDataset(train_items, base_dir, train_tf)
    val_ds = FoodDataset(val_items, base_dir, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_items, val_items

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels, _ in loader:
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

def run_experiment_1_head_only(base_dir: str, epochs: int = 15, lr: float = 1e-3) -> Dict[str, Any]:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: FREEZE BACKBONE & TRAIN 5-CLASS FOOD CLASSIFIER HEAD")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr}")
    print("=" * 70)

    train_loader, val_loader, train_items, val_items = create_dataloaders(base_dir)

    # Load pretrained model
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze features backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier output layer with 5 classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(FOOD_CLASSES))
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters (Head Only): {trainable_params:,}")
    print(f"Frozen Parameters (Backbone):     {frozen_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)

    history = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    ckpt_dir = os.path.join(base_dir, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "mobilenet_v3_food_head_only.pth")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for imgs, labels, _ in train_loader:
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

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4)
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_mapping": CLASS_TO_IDX,
                "experiment": "Experiment 1 (Head Only)"
            }, best_ckpt_path)
            saved_tag = "[BEST CHECKPOINT SAVED]"
        else:
            saved_tag = ""

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} {saved_tag}")

    elapsed = time.time() - t_start
    print(f"\nExperiment 1 Finished in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc:.2%}")
    return {
        "experiment_name": "Experiment 1 (Head Only)",
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 2),
        "best_val_acc": best_val_acc,
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint_path": "models/checkpoints/mobilenet_v3_food_head_only.pth",
        "history": history
    }

def run_experiment_2_unfreeze_upper(base_dir: str, epochs: int = 10, lr: float = 1e-4) -> Dict[str, Any]:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: STAGED FINE-TUNING (UNFREEZING UPPER BOTTLENECK BLOCKS)")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr}")
    print("=" * 70)

    train_loader, val_loader, _, _ = create_dataloaders(base_dir)

    # Initialize model with 5 classes
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(FOOD_CLASSES))

    # Load weights from Experiment 1 best checkpoint
    head_ckpt_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_head_only.pth")
    if not os.path.exists(head_ckpt_path):
        raise FileNotFoundError(f"Missing Experiment 1 checkpoint: {head_ckpt_path}")
    ckpt = torch.load(head_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # Freeze lower features, unfreeze top 2 blocks (features[11] and features[12]) + classifier
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[11].parameters():
        param.requires_grad = True
    for param in model.features[12].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters (Upper Blocks + Head): {trainable_params:,}")
    print(f"Frozen Parameters (Lower Backbone):       {frozen_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_finetuned.pth")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for imgs, labels, _ in train_loader:
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

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6)
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_mapping": CLASS_TO_IDX,
                "experiment": "Experiment 2 (Staged Fine-Tuned)"
            }, best_ckpt_path)
            saved_tag = "[BEST CHECKPOINT SAVED]"
        else:
            saved_tag = ""

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} {saved_tag}")

    elapsed = time.time() - t_start
    print(f"\nExperiment 2 Finished in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc:.2%}")
    return {
        "experiment_name": "Experiment 2 (Staged Fine-Tuned)",
        "epochs": epochs,
        "elapsed_sec": round(elapsed, 2),
        "best_val_acc": best_val_acc,
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint_path": "models/checkpoints/mobilenet_v3_food_finetuned.pth",
        "history": history
    }

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    res1 = run_experiment_1_head_only(base, epochs=15)
    res2 = run_experiment_2_unfreeze_upper(base, epochs=10)
    print("\nTraining complete.")
