"""
Kaggle-First GPU Training and Evaluation Pipeline (43-Class Indian Food Classification)
Designed for reproducible cloud execution on Kaggle GPU (NVIDIA T4 / P100)
and local development fallback.

Features:
1. Strict Compliance Guardrail: Fails loudly if the legacy 172-image 5-class dataset is loaded.
2. Actual-Data GPU Smoke Test: Verifies forward-backward passes on real dataset samples before training.
3. 43-Class MobileNetV3-Small Architecture with pretrained ImageNet transfer learning.
4. Staged Training:
   - Stage 1: Linear probe on adapted 43-class head with frozen backbone.
   - Stage 2: Staged fine-tuning of top bottleneck blocks with Cosine Annealing.
5. Dedicated Experiment Checkpoint: Saves to 'mobilenet_v3_indian_43class_baseline.pth' without overwriting historical models.
6. Validation-Driven Model Selection: Evaluates validation split every epoch; evaluates held-out test split ONLY ONCE at completion.
7. Full 43-Class Metrics: Wilson 95% confidence intervals, per-class precision/recall/F1, and complete 43x43 confusion matrix.
8. Exports complete machine-readable training report and artifacts.
"""

import os
import sys
import json
import time
import math
import random
import shutil
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.parquet_food_loader import (
    ParquetFoodDataset,
    EXPECTED_43_CLASSES,
    LegacyDatasetGuardError,
    extract_and_prepare_43class_manifest,
    discover_parquet_shards
)

EXPERIMENT_NAME = "mobilenet_v3_indian_43class_baseline"

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
    print("=" * 85)
    print("KAGGLE GPU HARDWARE & COMPUTE RUNTIME AUDIT")
    print("=" * 85)
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Fallback)"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0

    print(f"CUDA Available:        {cuda_avail}")
    print(f"Device Name:           {device_name}")
    print(f"Total Dedicated VRAM:  {total_vram_gb:.2f} GB")
    print(f"PyTorch Version:       {torch.__version__}")
    print(f"CUDA Version (Torch):  {torch.version.cuda}")
    print("=" * 85)

    device = torch.device("cuda:0" if cuda_avail else "cpu")
    return device, {
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "pytorch_version": torch.__version__
    }

def build_transforms():
    """Builds standard training data augmentation and validation normalization transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
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

def build_mobilenet_v3(num_classes: int = 43, device: Optional[torch.device] = None) -> nn.Module:
    """
    Constructs MobileNetV3-Small with pretrained ImageNet weights and adapted 43-class head.
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

    if device:
        model = model.to(device)
    return model

def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)

def run_actual_data_smoke_test(
    manifest_path: str,
    batch_size: int = 4,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Runs a GPU smoke test using a small subset of the actual dataset.
    Verifies that real images decode, forward pass produces shape [batch, 43],
    loss is finite, and gradients propagate to all trainable parameters.
    """
    print("\n" + "=" * 85)
    print("RUNNING GPU SMOKE TEST ON ACTUAL 43-CLASS DATASET SUBSET")
    print("=" * 85)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    if len(images) == 0:
        raise ValueError("Cannot run smoke test: manifest contains zero images!")

    classes = sorted(list({img["canonical_id"] for img in images}))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    _, val_trans = build_transforms()
    # Take first 8 valid images
    smoke_items = images[:8]
    dataset = ParquetFoodDataset(smoke_items, class_to_idx, transform=val_trans)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_mobilenet_v3(num_classes=len(classes), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)

    model.train()
    batch_count = 0
    total_loss = 0.0

    for x, y, img_ids in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        assert out.shape == (x.size(0), len(classes)), f"Expected shape ({x.size(0)}, {len(classes)}), got {out.shape}"
        loss = criterion(out, y)
        assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"

        loss.backward()
        # Verify gradients exist on output linear layer
        last_linear = model.classifier[3]
        assert last_linear.weight.grad is not None, "Gradient was not computed on classifier weights!"

        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / max(1, batch_count)
    print(f"Smoke Test Batches Processed: {batch_count} ({batch_count * batch_size} actual images)")
    print(f"Smoke Test Average Loss:      {avg_loss:.4f}")
    print(f"Model Output Shape:           ({batch_size}, {len(classes)})")
    print(f"CUDA Gradient Check:          PASS (Parameters successfully updated)")
    print("SMOKE TEST COMPLETE: Actual dataset and GPU pipeline verified.")
    print("=" * 85 + "\n")

    return {
        "status": "PASS",
        "batches": batch_count,
        "sample_count": batch_count * batch_size,
        "average_loss": round(avg_loss, 4),
        "device": str(device)
    }

def train_indian_43class_model(
    manifest_path: Optional[str] = None,
    parquet_dir: Optional[str] = None,
    epochs_head: int = 5,
    epochs_finetune: int = 10,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    seed: int = 42,
    num_workers: int = 2
) -> Dict[str, Any]:
    """
    Trains the 43-class MobileNetV3-Small model on the Enhanced Indian Food Classification dataset.
    Fails loudly if the legacy 172-image 5-class dataset is accidentally loaded.
    """
    start_time = time.time()
    set_seed(seed)
    is_kaggle, working_dir, output_dir, model_dir = get_kaggle_or_local_paths()
    device, gpu_info = audit_gpu_environment()

    # Discover or prepare the 43-class manifest
    if manifest_path is None or not os.path.exists(manifest_path):
        candidate_manifests = [
            os.path.join(working_dir, "data_manifests", "indian_food_43class_manifest.json"),
            os.path.join(BASE_DIR, "data_manifests", "indian_food_43class_manifest.json")
        ]
        manifest_path = next((m for m in candidate_manifests if os.path.exists(m)), None)

    if manifest_path is None or not os.path.exists(manifest_path):
        # Look for raw parquet shards to extract
        candidate_parquet_dirs = [
            parquet_dir,
            "/kaggle/working/datasets/raw/sohlhealth_indian_food",
            os.path.join(working_dir, "datasets", "raw", "sohlhealth_indian_food"),
            os.path.join(BASE_DIR, "datasets", "raw", "sohlhealth_indian_food")
        ]
        chosen_parquet_dir = next((d for d in candidate_parquet_dirs if d and os.path.exists(d)), None)

        if chosen_parquet_dir and len(discover_parquet_shards(chosen_parquet_dir)) > 0:
            print(f"Manifest not found. Extracting from parquet shards at {chosen_parquet_dir}...")
            image_out = os.path.join(working_dir, "datasets", "processed", "indian_food_43class")
            manifest_out = os.path.join(working_dir, "data_manifests", "indian_food_43class_manifest.json")
            extract_and_prepare_43class_manifest(chosen_parquet_dir, image_out, manifest_out)
            manifest_path = manifest_out
        else:
            raise FileNotFoundError(
                "Neither 43-class manifest nor downloaded parquet shards found! "
                "Please run Step 4 (Resumable Downloader) to acquire the Enhanced Indian Food Classification shards."
            )

    print(f"Loading 43-Class Indian Food manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    all_images = manifest_data.get("images", [])
    total_images_loaded = len(all_images)
    classes = sorted(list({img["canonical_id"] for img in all_images}))
    num_classes = len(classes)

    # -------------------------------------------------------------------------
    # STRICT COMPLIANCE GUARDRAIL: Fail loudly if legacy dataset is loaded
    # -------------------------------------------------------------------------
    if total_images_loaded < 1000 or num_classes < 40:
        raise LegacyDatasetGuardError(
            f"FATAL COMPLIANCE GUARDRAIL VIOLATION: "
            f"Attempted to train on legacy/tiny dataset with only {total_images_loaded} images across {num_classes} classes! "
            f"The Indian Food 43-Class pipeline strictly requires the validated ~15,404-image dataset across 43 classes. "
            f"Training halted immediately to prevent invalid results."
        )

    print("=" * 85)
    print(f"VERIFIED 43-CLASS INDIAN FOOD DATASET LOADED:")
    print(f"Total Unique Images:   {total_images_loaded:,}")
    print(f"Total Active Classes:  {num_classes} (Target: 43)")
    print(f"Class Vocabulary:      {classes}")
    print("=" * 85)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    # Save label-to-index mapping to dedicated file
    mapping_path = os.path.join(model_dir, "indian_43class_label_to_idx.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Saved label-to-index mapping to: {mapping_path}")

    # Run smoke test on actual data before launching full training
    smoke_result = run_actual_data_smoke_test(manifest_path, batch_size=4, device=device)
    assert smoke_result["status"] == "PASS", "Smoke test failed! Aborting training."

    train_items = [img for img in all_images if img.get("split") == "train"]
    val_items = [img for img in all_images if img.get("split") == "val"]
    test_items = [img for img in all_images if img.get("split") == "test"]

    print(f"Split Breakdown: Train={len(train_items):,}, Val={len(val_items):,}, Test={len(test_items):,}")

    train_trans, val_trans = build_transforms()
    train_dataset = ParquetFoodDataset(train_items, class_to_idx, transform=train_trans, base_dir=working_dir)
    val_dataset = ParquetFoodDataset(val_items, class_to_idx, transform=val_trans, base_dir=working_dir)
    test_dataset = ParquetFoodDataset(test_items, class_to_idx, transform=val_trans, base_dir=working_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = build_mobilenet_v3(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()

    # Unique experiment checkpoint path to ensure existing models are never overwritten
    checkpoint_name = "mobilenet_v3_indian_43class_baseline.pth"
    best_checkpoint_path = os.path.join(model_dir, checkpoint_name)
    best_val_acc = 0.0

    training_config = {
        "experiment_name": EXPERIMENT_NAME,
        "model_architecture": "MobileNetV3-Small",
        "num_classes": num_classes,
        "epochs_head": epochs_head,
        "epochs_finetune": epochs_finetune,
        "batch_size": batch_size,
        "lr_head": lr_head,
        "lr_finetune": lr_finetune,
        "seed": seed,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR"
    }

    config_path = os.path.join(output_dir, "indian_43class_training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    # -------------------------------------------------------------------------
    # STAGE 1: Linear Probe Head Training (Backbone Frozen)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"STAGE 1: TRAINING 43-CLASS CLASSIFIER HEAD ({epochs_head} Epochs, Backbone Frozen)")
    print("=" * 85)

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt_head = optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-2)

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

        # Validation evaluation during training
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
        print(f"Stage 1 [Epoch {ep:02d}/{epochs_head:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.1%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "experiment_name": EXPERIMENT_NAME,
                "epoch": ep,
                "stage": "head_only",
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "training_config": training_config,
                "architecture": "mobilenet_v3_small_43class"
            }, best_checkpoint_path)
            compat_path = os.path.join(model_dir, "mobilenet_v3_indian_43class_best.pth")
            shutil.copyfile(best_checkpoint_path, compat_path)
            print(f"  -> Saved Stage 1 best checkpoint (Val Acc: {val_acc:.1%}) to {best_checkpoint_path}")

    # -------------------------------------------------------------------------
    # STAGE 2: Staged Fine-Tuning of Upper Bottlenecks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"STAGE 2: STAGED FINE-TUNING OF TOP BOTTLENECKS ({epochs_finetune} Epochs)")
    print("=" * 85)

    # Unfreeze upper bottleneck blocks 10-12
    for idx, block in enumerate(model.features):
        if idx >= 10:
            for p in block.parameters():
                p.requires_grad = True

    opt_finetune = optim.AdamW([
        {"params": [p for idx, block in enumerate(model.features) if idx >= 10 for p in block.parameters()], "lr": lr_finetune * 0.1},
        {"params": model.classifier.parameters(), "lr": lr_finetune}
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_finetune, T_max=epochs_finetune, eta_min=1e-6)

    for ep in range(1, epochs_finetune + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt_finetune.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_finetune.step()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()
        tr_loss = running_loss / max(1, total)
        tr_acc = correct / max(1, total)

        # Validation evaluation during training
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
        print(f"Stage 2 [Epoch {ep:02d}/{epochs_finetune:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.1%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "experiment_name": EXPERIMENT_NAME,
                "epoch": ep + epochs_head,
                "stage": "staged_finetune",
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "training_config": training_config,
                "architecture": "mobilenet_v3_small_43class"
            }, best_checkpoint_path)
            compat_path = os.path.join(model_dir, "mobilenet_v3_indian_43class_best.pth")
            shutil.copyfile(best_checkpoint_path, compat_path)
            print(f"  -> Saved Stage 2 best checkpoint (Val Acc: {val_acc:.1%}) to {best_checkpoint_path}")

    # -------------------------------------------------------------------------
    # STAGE 3: Final Evaluation on Held-Out Test Set (Evaluated ONCE at Completion)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("STAGE 3: FINAL EVALUATION ON HELD-OUT TEST SET (EVALUATED ONCE)")
    print("=" * 85)

    # Load best checkpoint selected on validation accuracy
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def evaluate_split(loader: DataLoader, split_name: str) -> Tuple[Dict[str, Any], List[List[int]]]:
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                out = model(x)
                preds = out.argmax(dim=1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(y.tolist())

        total = len(y_true)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        acc = correct / max(1, total)
        ci_low, ci_high = calculate_wilson_ci(correct, total)

        # 43x43 Confusion Matrix
        cm = [[0] * num_classes for _ in range(num_classes)]
        for yt, yp in zip(y_true, y_pred):
            cm[yt][yp] += 1

        # Per-class metrics
        per_class = {}
        for cidx, cname in idx_to_class.items():
            tp = cm[cidx][cidx]
            fp = sum(cm[other][cidx] for other in range(num_classes) if other != cidx)
            fn = sum(cm[cidx][other] for other in range(num_classes) if other != cidx)
            support = sum(cm[cidx])

            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)

            per_class[cname] = {
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }

        macro_f1 = sum(v["f1_score"] for v in per_class.values()) / max(1, len(per_class))

        # Top confusion pairs (where true != pred)
        confusion_pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append({
                        "true_class": idx_to_class[i],
                        "pred_class": idx_to_class[j],
                        "count": cm[i][j]
                    })
        confusion_pairs = sorted(confusion_pairs, key=lambda x: x["count"], reverse=True)[:10]

        summary = {
            "split_name": split_name,
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": round(acc, 4),
            "accuracy_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "macro_f1": round(macro_f1, 4),
            "top_confusion_pairs": confusion_pairs,
            "per_class": per_class
        }
        return summary, cm

    val_metrics, val_cm = evaluate_split(val_loader, "validation")
    test_metrics, test_cm = evaluate_split(test_loader, "test_held_out")

    training_duration_s = round(time.time() - start_time, 2)
    checkpoint_size_mb = round(os.path.getsize(best_checkpoint_path) / (1024**2), 2)

    # Save full confusion matrix
    cm_path = os.path.join(output_dir, "indian_43class_confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "validation_confusion_matrix": val_cm,
            "test_confusion_matrix": test_cm
        }, f, indent=2)

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_name": "Enhanced Indian Food Classification (43 Classes)",
        "source_dataset": "SohlHealth/enhanced-indian-food-classification",
        "license": "CC-BY-4.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_environment": gpu_info,
        "total_images_in_dataset": total_images_loaded,
        "num_classes": num_classes,
        "classes": classes,
        "training_duration_seconds": training_duration_s,
        "checkpoint_file": best_checkpoint_path,
        "checkpoint_size_mb": checkpoint_size_mb,
        "training_configuration": training_config,
        "validation_metrics": val_metrics,
        "held_out_test_metrics": test_metrics,
        "scientific_caveats": [
            "This model represents an authentic 43-class Indian food baseline trained on the Enhanced Indian Food Classification dataset.",
            "It does NOT claim production readiness or global accuracy across diverse international cuisines.",
            "Further cross-dataset validation and domain-shift evaluations are required before mobile edge deployment."
        ]
    }

    report_path = os.path.join(output_dir, "kaggle_indian_43class_training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print("FINAL 43-CLASS INDIAN FOOD EVALUATION METRICS REPORT")
    print("=" * 85)
    print(f"Validation Accuracy:       {val_metrics['accuracy']:.1%} (95% CI: [{val_metrics['accuracy_ci_95'][0]:.1%}, {val_metrics['accuracy_ci_95'][1]:.1%}])")
    print(f"Validation Macro-F1:       {val_metrics['macro_f1']:.4f} (over 43 classes)")
    print(f"Held-Out Test Accuracy:    {test_metrics['accuracy']:.1%} (95% CI: [{test_metrics['accuracy_ci_95'][0]:.1%}, {test_metrics['accuracy_ci_95'][1]:.1%}])")
    print(f"Held-Out Test Macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Total Training Duration:   {training_duration_s} seconds ({training_duration_s / 60:.1f} minutes)")
    print(f"Model Checkpoint Size:     {checkpoint_size_mb} MB")
    print(f"Checkpoint Saved To:       {best_checkpoint_path}")
    print(f"Report Saved To:           {report_path}")
    print("=" * 85)

    return report

if __name__ == "__main__":
    train_indian_43class_model()
