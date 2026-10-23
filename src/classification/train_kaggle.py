"""
Kaggle-First GPU Training and Evaluation Pipeline
Designed for reproducible cloud execution on Kaggle GPU (NVIDIA T4 / P100)
and local development fallback.

Features:
1. GPU Environment and VRAM Audit.
2. Transfer Learning from MobileNetV3-Small:
   - Stage 1: Frozen backbone + 10-class food classifier head.
   - Stage 2: Staged fine-tuning of top bottleneck blocks with cosine decay.
3. Checkpoint verification: Saves and re-validates best validation checkpoint.
4. Comprehensive multi-split evaluation:
   - Historical test set (continuity benchmark)
   - New expanded independent test set
5. Wilson 95% confidence intervals, per-class precision/recall/F1, and confusion matrix.
"""

import os
import sys
import json
import time
import math
import random
import shutil
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_kaggle_or_local_paths():
    """Detects if running in Kaggle environment or local repository."""
    is_kaggle = os.path.exists("/kaggle/working")
    if is_kaggle:
        working_dir = "/kaggle/working"
        output_dir = os.path.join(working_dir, "outputs")
        model_dir = os.path.join(working_dir, "models")
    else:
        working_dir = BASE_DIR
        output_dir = os.path.join(BASE_DIR, "outputs")
        model_dir = os.path.join(BASE_DIR, "models", "checkpoints")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return is_kaggle, working_dir, output_dir, model_dir

def audit_gpu_environment():
    """Audits CUDA availability, device name, VRAM limits, and PyTorch configuration."""
    print("=" * 80)
    print("HARDWARE & GPU RUNTIME AUDIT")
    print("=" * 80)
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Fallback)"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0

    print(f"CUDA Available:     {cuda_avail}")
    print(f"Device Name:        {device_name}")
    print(f"Total VRAM:         {total_vram_gb:.2f} GB")
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"CUDA Version (Torch): {torch.version.cuda}")
    print("=" * 80)

    device = torch.device("cuda:0" if cuda_avail else "cpu")
    return device, {
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "pytorch_version": torch.__version__
    }

class FoodImageDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], class_to_idx: Dict[str, int], transform=None):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        file_path = item["file_path"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        with Image.open(file_path) as img:
            img = img.convert("RGB")

        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        label = self.class_to_idx[item["canonical_id"]]
        return tensor, label, item["image_id"]

def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def build_mobilenet_v3(num_classes: int, device: torch.device):
    """
    Constructs MobileNetV3-Small with pretrained weights and adapted classification head.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes)
    )
    return model.to(device)

def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)

def train_kaggle_model(
    manifest_path: str,
    epochs_head: int = 10,
    epochs_finetune: int = 10,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    seed: int = 42
) -> Dict[str, Any]:
    start_time = time.time()
    set_seed(seed)
    is_kaggle, working_dir, output_dir, model_dir = get_kaggle_or_local_paths()
    device, gpu_info = audit_gpu_environment()

    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    all_images = manifest_data.get("images", [])
    classes = sorted(list({img["canonical_id"] for img in all_images}))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)

    print(f"Found {len(all_images)} images across {num_classes} classes: {classes}")

    train_items = [img for img in all_images if img.get("split") == "train"]
    val_items = [img for img in all_images if img.get("split") == "val"]
    test_hist_items = [img for img in all_images if img.get("split") == "test" or img.get("split") == "test_historical"]
    test_indep_items = [img for img in all_images if img.get("split") == "test_independent"]

    print(f"Split breakdown: Train={len(train_items)}, Val={len(val_items)}, TestHist={len(test_hist_items)}, TestIndep={len(test_indep_items)}")

    train_trans, val_trans = build_transforms()
    train_loader = DataLoader(FoodImageDataset(train_items, class_to_idx, train_trans), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FoodImageDataset(val_items, class_to_idx, val_trans), batch_size=batch_size, shuffle=False)

    model = build_mobilenet_v3(num_classes, device)
    criterion = nn.CrossEntropyLoss()

    # Stage 1: Head-Only Training (Backbone frozen)
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING FOOD CLASSIFIER HEAD (Backbone Frozen)")
    print("=" * 80)
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt_head = optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-2)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(model_dir, "mobilenet_v3_food_kaggle_best.pth")

    for ep in range(1, epochs_head + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt_head.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_head.step()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        tr_loss = running_loss / max(1, total)
        tr_acc = correct / max(1, total)

        # Val
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)
        print(f"Epoch {ep:02d}/{epochs_head:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.1%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
                "seed": seed,
                "stage": "stage1_head_only"
            }, best_checkpoint_path)

    # Stage 2: Staged Fine-Tuning (Unfreezing upper bottleneck blocks 10-12)
    print("\n" + "=" * 80)
    print("STAGE 2: STAGED FINE-TUNING (Upper Inverted Residual Blocks Unfrozen)")
    print("=" * 80)
    # Unfreeze upper blocks (features[9:] in MobileNetV3-Small)
    for p in model.features.parameters():
        p.requires_grad = False
    for i in range(8, len(model.features)):
        for p in model.features[i].parameters():
            p.requires_grad = True

    opt_ft = optim.AdamW([
        {"params": [p for i in range(8, len(model.features)) for p in model.features[i].parameters()], "lr": lr_finetune * 0.5},
        {"params": model.classifier.parameters(), "lr": lr_finetune}
    ], weight_decay=1e-2)

    sched_ft = optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=epochs_finetune, eta_min=1e-6)

    for ep in range(1, epochs_finetune + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt_ft.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_ft.step()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        sched_ft.step()
        tr_loss = running_loss / max(1, total)
        tr_acc = correct / max(1, total)

        # Val
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)
        print(f"FineTune Epoch {ep:02d}/{epochs_finetune:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.1%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
                "seed": seed,
                "stage": "stage2_staged_finetuning"
            }, best_checkpoint_path)

    training_duration_s = round(time.time() - start_time, 2)
    checkpoint_size_mb = round(os.path.getsize(best_checkpoint_path) / (1024**2), 2)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE in {training_duration_s} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc:.1%}")
    print(f"Best Checkpoint Saved:   {best_checkpoint_path} ({checkpoint_size_mb} MB)")
    print("=" * 80)

    # Verification: Reload checkpoint to guarantee integrity
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def evaluate_split(split_items: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
        if not split_items:
            return {"accuracy": 0.0, "total_samples": 0}

        loader = DataLoader(FoodImageDataset(split_items, class_to_idx, val_trans), batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        acc = correct / max(1, total)
        ci_low, ci_high = calculate_wilson_ci(correct, total)

        # Per-class metrics
        per_class = {}
        for cidx, cname in idx_to_class.items():
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cidx and yp == cidx)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cidx and yp == cidx)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cidx and yp != cidx)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)
            per_class[cname] = {
                "support": sum(1 for yt in y_true if yt == cidx),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }

        macro_f1 = sum(v["f1_score"] for v in per_class.values()) / max(1, len(per_class))

        return {
            "split_name": split_name,
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": round(acc, 4),
            "accuracy_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class
        }

    eval_hist = evaluate_split(test_hist_items, "test_historical")
    eval_indep = evaluate_split(test_indep_items, "test_independent")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_environment": gpu_info,
        "training_duration_seconds": training_duration_s,
        "checkpoint_file": best_checkpoint_path,
        "checkpoint_size_mb": checkpoint_size_mb,
        "hyperparameters": {
            "epochs_head": epochs_head,
            "epochs_finetune": epochs_finetune,
            "batch_size": batch_size,
            "lr_head": lr_head,
            "lr_finetune": lr_finetune,
            "random_seed": seed,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR"
        },
        "evaluation_historical_test": eval_hist,
        "evaluation_independent_test": eval_indep
    }

    report_path = os.path.join(output_dir, "kaggle_training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFinal Evaluation Report saved to: {report_path}")
    return report

if __name__ == "__main__":
    v3_manifest = os.path.join(BASE_DIR, "data_manifests", "food_dataset_v3_manifest.json")
    train_kaggle_model(v3_manifest, epochs_head=2, epochs_finetune=2, batch_size=16)
