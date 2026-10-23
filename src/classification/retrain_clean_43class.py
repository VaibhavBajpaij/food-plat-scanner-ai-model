"""
Clean Retraining Pipeline for 43-Class Indian Food Classification
Uses MobileNetV3-Small starting strictly from official ImageNet-pretrained weights
(NOT the contaminated checkpoint).

Requirements & Constraints:
1. Starts from models.MobileNet_V3_Small_Weights.DEFAULT.
2. Trains ONLY on the repaired training split from indian_food_43class_repaired_manifest_v2.json.
3. Uses the repaired validation split for model selection and early stopping.
4. Strictly forbids using the held-out test split for training or model selection.
5. Verifies duplicate-group isolation and label mapping before training.
6. Verifies 100% of referenced training and validation images exist on disk before training.
7. If images are missing, safely stops and explains precisely how to restore/attach them on Kaggle.
8. Saves resumable checkpoints (latest and best) and packaging zip with model config, labels, and reports.
9. Evaluates held-out test split exactly once after model selection.
10. Supports resume from interrupted checkpoints via --resume.
"""

import os
import sys
import io
import json
import time
import math
import random
import shutil
import zipfile
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# The 43 canonical classes of the Enhanced Indian Food Classification dataset
CANONICAL_43_FOOD_CLASSES = sorted([
    "aloo_gobi", "aloo_masala", "bhindi_masala", "biryani", "burger",
    "butter_naan", "chai", "chapati", "chicken_curry", "chole",
    "chole_bhature", "coconut_chutney", "dal_makhani", "dhokla", "dum_aloo",
    "fish_curry", "fried_rice", "ghevar", "green_chutney", "gulab_jamun",
    "idli", "jalebi", "kaathi_rolls", "kadai_paneer", "kebab",
    "kheer", "kulfi", "lassi", "masala_dosa", "modak",
    "momos", "mutton_curry", "paani_puri", "pakode", "palak_paneer",
    "pav_bhaji", "pizza", "poha", "rajma_curry", "ras_malai",
    "samosa", "shahi_paneer", "white_rice"
])

EXPERIMENT_NAME = "mobilenet_v3_indian_43class_clean_v2"

class DatasetPathValidationError(Exception):
    """Raised when dataset image files referenced in the manifest do not exist on disk."""
    pass

class SplitLeakageError(Exception):
    """Raised when duplicate groups or test items cross split boundaries."""
    pass

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    if k == 0:
        return 0.0, min(1.0, centre + spread)
    if k == n:
        return max(0.0, centre - spread), 1.0
    return max(0.0, centre - spread), min(1.0, centre + spread)

def build_clean_mobilenet_v3(
    num_classes: int = 43,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Constructs MobileNetV3-Small with official ImageNet pretrained weights.
    Strictly avoids loading weights from previously fine-tuned checkpoints.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[0].in_features  # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes)
    )

    if device:
        model = model.to(device)
    return model

def verify_label_mapping(manifest_classes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Verifies that the manifest classes conform to the 43 expected canonical classes."""
    sorted_manifest = sorted(manifest_classes)
    if sorted_manifest != CANONICAL_43_FOOD_CLASSES:
        diff_missing = set(CANONICAL_43_FOOD_CLASSES) - set(sorted_manifest)
        diff_extra = set(sorted_manifest) - set(CANONICAL_43_FOOD_CLASSES)
        raise ValueError(
            f"Label mapping mismatch! Missing: {diff_missing}, Extra: {diff_extra}"
        )
    class_to_idx = {c: i for i, c in enumerate(CANONICAL_43_FOOD_CLASSES)}
    idx_to_class = {i: c for i, c in enumerate(CANONICAL_43_FOOD_CLASSES)}
    return class_to_idx, idx_to_class

def verify_duplicate_group_isolation(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verifies that every group_id belongs to exactly one split.
    Guarantees zero cross-split leakage of duplicate or near-duplicate images.
    """
    group_splits = defaultdict(set)
    for img in images:
        gid = img.get("group_id", img.get("image_id"))
        group_splits[gid].add(img["split"])

    leaking_groups = {gid: sp for gid, sp in group_splits.items() if len(sp) > 1}
    if leaking_groups:
        raise SplitLeakageError(
            f"FATAL SPLIT LEAKAGE: {len(leaking_groups)} duplicate groups cross split boundaries! "
            f"Example: {list(leaking_groups.items())[:3]}"
        )
    return {
        "total_groups": len(group_splits),
        "leaking_groups_count": 0,
        "is_isolated": True
    }

def resolve_image_path(path: str, base_dir: str = BASE_DIR) -> Optional[str]:
    """
    Resolves an image path across local workspace, Kaggle /kaggle/working,
    and attached Kaggle datasets in /kaggle/input.
    """
    if os.path.exists(path):
        return path

    candidates = [
        os.path.join(base_dir, path),
        os.path.join(base_dir, "datasets", os.path.basename(path)),
        os.path.join(os.path.dirname(base_dir), path)
    ]

    norm = path.replace("\\", "/")
    if "datasets/" in norm:
        suffix = norm[norm.index("datasets/"):]
        candidates.extend([
            os.path.join(base_dir, suffix),
            os.path.join(os.path.dirname(base_dir), suffix),
            os.path.join("/kaggle/working", suffix),
            os.path.join("/kaggle/working/food-plat-scanner-ai-model", suffix)
        ])

    if "indian_food_43class/" in norm:
        rel_under_dataset = norm[norm.index("indian_food_43class/") + len("indian_food_43class/"):].lstrip("/")
        candidates.extend([
            os.path.join(base_dir, "datasets", "processed", "indian_food_43class", rel_under_dataset),
            os.path.join("/kaggle/working/datasets/processed/indian_food_43class", rel_under_dataset),
        ])

    # Search inside /kaggle/input if present
    if os.path.exists("/kaggle/input"):
        try:
            for sub in os.listdir("/kaggle/input"):
                input_sub = os.path.join("/kaggle/input", sub)
                if os.path.isdir(input_sub):
                    if "datasets/" in norm:
                        candidates.append(os.path.join(input_sub, norm[norm.index("datasets/"):]))
                    if "indian_food_43class/" in norm:
                        candidates.append(os.path.join(input_sub, rel_under_dataset))
                        candidates.append(os.path.join(input_sub, "indian_food_43class", rel_under_dataset))
                    # Also check class/basename directly
                    parts = norm.split("/")
                    if len(parts) >= 2:
                        candidates.append(os.path.join(input_sub, parts[-2], parts[-1]))
                    candidates.append(os.path.join(input_sub, os.path.basename(path)))
        except Exception:
            pass

    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def verify_dataset_paths(
    images: List[Dict[str, Any]],
    base_dir: str = BASE_DIR,
    splits_to_verify: Optional[List[str]] = None
) -> Tuple[bool, int, int, List[str]]:
    """
    Comprehensive path verification: checks if physical image files exist on disk.
    If splits_to_verify is None, checks all images.
    Returns (all_exist, total_checked, missing_count, sample_missing_paths).
    """
    if splits_to_verify is not None:
        target_splits = set(splits_to_verify)
        images_to_check = [img for img in images if img.get("split") in target_splits]
    else:
        images_to_check = images

    total_checked = len(images_to_check)
    missing = []

    for img in images_to_check:
        raw_path = img.get("file_path", "")
        resolved = resolve_image_path(raw_path, base_dir=base_dir)
        if not resolved:
            missing.append(raw_path)

    all_exist = len(missing) == 0
    return all_exist, total_checked, len(missing), missing[:5]

class ManifestFoodDataset(Dataset):
    """
    PyTorch Dataset reading from manifest items with safe path resolution.
    Strictly rejects items not belonging to the intended split.
    """
    def __init__(
        self,
        items: List[Dict[str, Any]],
        class_to_idx: Dict[str, int],
        expected_split: str,
        transform=None,
        base_dir: str = BASE_DIR
    ):
        self.expected_split = expected_split
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.base_dir = base_dir

        # Guardrail: Verify split assignment
        self.items = []
        for it in items:
            if it["split"] != expected_split:
                raise SplitLeakageError(
                    f"Dataset configured for split '{expected_split}' received an item from split '{it['split']}'! "
                    f"Item ID: {it.get('image_id')}"
                )
            self.items.append(it)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        resolved = resolve_image_path(item["file_path"], base_dir=self.base_dir)
        if not resolved:
            raise FileNotFoundError(f"Image not found at any candidate location: {item['file_path']}")

        with Image.open(resolved) as img:
            img = img.convert("RGB")

        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        label_idx = self.class_to_idx[item["canonical_id"]]
        return tensor, label_idx, item["image_id"]

class SyntheticSmokeDataset(Dataset):
    """
    Deterministic in-memory synthetic dataset for local end-to-end verification
    when full dataset images are stored in cloud/Kaggle environment.
    """
    def __init__(
        self,
        num_samples_per_class: int = 2,
        classes: Optional[List[str]] = None,
        transform=None
    ):
        self.classes = classes or CANONICAL_43_FOOD_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.samples = []

        rng = random.Random(42)
        for cidx, cname in enumerate(self.classes):
            for s in range(num_samples_per_class):
                color = (
                    (cidx * 37 + s * 50) % 256,
                    (cidx * 73 + s * 30) % 256,
                    (cidx * 109 + s * 70) % 256
                )
                self.samples.append((cname, cidx, color, f"synth_{cname}_{s}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        cname, cidx, color, sample_id = self.samples[idx]
        img = Image.new("RGB", (224, 224), color=color)
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        return tensor, cidx, sample_id

def build_data_transforms():
    """Standard ImageNet normalization and data augmentation."""
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    eval_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return train_trans, eval_trans

def evaluate_dataloader(
    model: nn.Module,
    loader: DataLoader,
    classes: List[str],
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates model on dataloader, computing:
    - Loss & Accuracy
    - Balanced Accuracy
    - Macro-F1 & Weighted-F1
    - Per-class metrics
    - 43x43 Confusion Matrix
    - Wilson 95% CI
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_classes = len(classes)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)

            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    total = len(all_targets)
    if total == 0:
        return {"accuracy": 0.0, "total": 0}

    avg_loss = total_loss / total
    correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    acc = correct / total
    ci_low, ci_high = calculate_wilson_ci(correct, total)

    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    per_class = {}
    class_recalls = []
    weighted_f1_sum = 0.0

    for cidx, cname in enumerate(classes):
        tp = cm[cidx][cidx]
        fp = sum(cm[other][cidx] for other in range(num_classes) if other != cidx)
        fn = sum(cm[cidx][other] for other in range(num_classes) if other != cidx)
        support = sum(cm[cidx])

        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

        if support > 0:
            class_recalls.append(rec)
            weighted_f1_sum += f1 * support

        per_class[cname] = {
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        }

    macro_f1 = sum(v["f1_score"] for v in per_class.values()) / max(1, num_classes)
    balanced_acc = sum(class_recalls) / max(1, len(class_recalls)) if class_recalls else 0.0
    weighted_f1 = weighted_f1_sum / max(1, total)

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "accuracy_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm
    }

def package_artifacts_zip(
    checkpoint_best_path: str,
    label_mapping_path: str,
    model_config_path: str,
    training_report_path: str,
    manifest_path: str,
    zip_output_path: str
) -> str:
    """
    Packages all training deliverables into a clean, standalone ZIP file.
    """
    os.makedirs(os.path.dirname(zip_output_path), exist_ok=True)
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(checkpoint_best_path):
            zipf.write(checkpoint_best_path, arcname=os.path.basename(checkpoint_best_path))
        latest_ckpt = checkpoint_best_path.replace("_best.pth", "_latest.pth").replace(".pth", "_latest.pth")
        if os.path.exists(latest_ckpt):
            zipf.write(latest_ckpt, arcname=os.path.basename(latest_ckpt))
        if os.path.exists(label_mapping_path):
            zipf.write(label_mapping_path, arcname=os.path.basename(label_mapping_path))
        if os.path.exists(model_config_path):
            zipf.write(model_config_path, arcname=os.path.basename(model_config_path))
        if os.path.exists(training_report_path):
            zipf.write(training_report_path, arcname=os.path.basename(training_report_path))
        if os.path.exists(manifest_path):
            zipf.write(manifest_path, arcname=os.path.basename(manifest_path))

    # Also duplicate to /kaggle/working if in Kaggle
    if os.path.exists("/kaggle/working"):
        kaggle_dest = os.path.join("/kaggle/working", os.path.basename(zip_output_path))
        if os.path.abspath(zip_output_path) != os.path.abspath(kaggle_dest):
            shutil.copyfile(zip_output_path, kaggle_dest)

    return zip_output_path

def run_clean_retraining(
    manifest_path: str = "data_manifests/indian_food_43class_repaired_manifest_v2.json",
    output_checkpoint_path: str = "models/checkpoints/mobilenet_v3_indian_43class_clean_v2_best.pth",
    output_report_path: str = "outputs/indian_43class_clean_training_report.json",
    epochs_head: int = 5,
    epochs_finetune: int = 10,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    seed: int = 42,
    smoke_mode: bool = False,
    resume_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes clean retraining pipeline from ImageNet pretrained weights.
    Strictly verifies duplicate-group isolation, class mapping, and dataset paths.
    """
    start_time = time.time()
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("CLEAN RETRAINING: 43-CLASS INDIAN FOOD CLASSIFIER")
    print(f"Device: {device} | Pretrained Base: MobileNetV3-Small (Official ImageNet Weights)")
    print(f"Seed: {seed} | Manifest: {manifest_path}")
    print("=" * 85)

    # 1. Load manifest and verify mapping
    if not os.path.exists(manifest_path):
        # Look in candidate locations
        alt_paths = [
            os.path.join(BASE_DIR, manifest_path),
            os.path.join("/kaggle/working/food-plat-scanner-ai-model", manifest_path),
            os.path.join("/kaggle/working", manifest_path)
        ]
        found = next((p for p in alt_paths if os.path.exists(p)), None)
        if found:
            manifest_path = found
        else:
            raise FileNotFoundError(f"Repaired manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    classes = manifest_data.get("classes", [])
    class_to_idx, idx_to_class = verify_label_mapping(classes)
    num_classes = len(classes)
    print(f"[OK] Label mapping verified: exactly {num_classes} canonical classes.")

    # 2. Verify duplicate-group isolation
    images = manifest_data.get("images", [])
    group_isolation = verify_duplicate_group_isolation(images)
    print(f"[OK] Duplicate-group isolation verified: {group_isolation['total_groups']} groups with zero cross-split leakage.")

    # 3. Path verification: Every referenced train and val image must exist on disk
    print("Verifying that every referenced training and validation image exists on disk...")
    paths_ok, total_chk, missing_cnt, missing_samples = verify_dataset_paths(
        images,
        base_dir=BASE_DIR,
        splits_to_verify=["train", "val"]
    )

    if not paths_ok and not smoke_mode:
        print("\n" + "!" * 85)
        print("DATASET PATH VALIDATION REPORT: MISSING PHYSICAL FILES")
        print("!" * 85)
        print(f"Missing {missing_cnt:,} of {total_chk:,} referenced training/validation images!")
        print(f"Sample missing path: {missing_samples[0] if missing_samples else 'N/A'}")
        print("\nDIAGNOSIS & RESTORATION GUIDE FOR KAGGLE:")
        print("  Option 1 (Automated Download & Extraction in Kaggle Session):")
        print("    1. Run Step 4 in 'kaggle_starter.ipynb' to download the 5 shards and extract images to '/kaggle/working/datasets/processed/indian_food_43class'.")
        print("    2. Run Step 5 in 'kaggle_starter.ipynb' to verify the repaired manifest and physical files.")
        print("  Option 2 (Attach Existing Kaggle Dataset):")
        print("    1. In Kaggle notebook, click '+ Add Input' -> search and attach the dataset.")
        print("    2. The images will automatically be discovered in '/kaggle/input'.")
        print("\nPer project instructions: Stopping safely without creating fake images or claiming success.")
        print("!" * 85 + "\n")
        raise DatasetPathValidationError(
            f"Physical image files missing on disk ({missing_cnt} missing of {total_chk} checked). "
            f"Run Step 4 in the Kaggle notebook to download and extract dataset images."
        )
    elif paths_ok:
        print(f"[OK] All {total_chk:,} referenced training and validation images verified on disk.")

    # 4. Save clean initialization weights & configuration
    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
    init_model = build_clean_mobilenet_v3(num_classes=num_classes, pretrained=True, device=device)
    init_ckpt_path = output_checkpoint_path.replace(".pth", "_init.pth").replace("_best", "")
    torch.save({
        "architecture": "mobilenet_v3_small_43class",
        "num_classes": num_classes,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "model_state_dict": init_model.state_dict(),
        "weights_source": "torchvision.models.MobileNet_V3_Small_Weights.DEFAULT",
        "is_clean_initialization": True,
        "note": "Clean ImageNet-pretrained weights with adapted 43-class head. Zero contamination from old food training."
    }, init_ckpt_path)
    print(f"[OK] Saved clean initialization checkpoint to: {init_ckpt_path}")

    # Save label-to-idx mapping
    model_dir = os.path.dirname(output_checkpoint_path)
    label_mapping_path = os.path.join(model_dir, "indian_43class_label_to_idx.json")
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    # Save model configuration
    model_config_path = os.path.join(model_dir, "model_config.json")
    model_config = {
        "experiment_name": EXPERIMENT_NAME,
        "base_architecture": "mobilenet_v3_small",
        "pretrained_weights": "torchvision.models.MobileNet_V3_Small_Weights.DEFAULT",
        "num_classes": num_classes,
        "classifier_head": "Linear(576, 1024) -> Hardswish() -> Dropout(0.2) -> Linear(1024, 43)",
        "dataset_name": "Enhanced Indian Food Classification (43 Classes - Repaired Split v2)",
        "manifest_path": manifest_path,
        "epochs_head": epochs_head,
        "epochs_finetune": epochs_finetune,
        "batch_size": batch_size,
        "lr_head": lr_head,
        "lr_finetune": lr_finetune,
        "seed": seed
    }
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # 5. Build Datasets
    train_trans, eval_trans = build_data_transforms()

    if smoke_mode:
        print("\n--- RUNNING IN SMOKE VERIFICATION MODE (In-Memory 43-Class Dataset) ---")
        train_ds = SyntheticSmokeDataset(num_samples_per_class=3, classes=classes, transform=train_trans)
        val_ds = SyntheticSmokeDataset(num_samples_per_class=2, classes=classes, transform=eval_trans)
        test_ds = SyntheticSmokeDataset(num_samples_per_class=2, classes=classes, transform=eval_trans)
        eff_epochs_head = 2
        eff_epochs_finetune = 2
        batch_size = 16
    else:
        train_items = [img for img in images if img["split"] == "train"]
        val_items = [img for img in images if img["split"] == "val"]
        test_items = [img for img in images if img["split"] == "test"]

        train_ds = ManifestFoodDataset(train_items, class_to_idx, expected_split="train", transform=train_trans, base_dir=BASE_DIR)
        val_ds = ManifestFoodDataset(val_items, class_to_idx, expected_split="val", transform=eval_trans, base_dir=BASE_DIR)
        test_ds = ManifestFoodDataset(test_items, class_to_idx, expected_split="test", transform=eval_trans, base_dir=BASE_DIR)
        eff_epochs_head = epochs_head
        eff_epochs_finetune = epochs_finetune

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    print(f"Data Partitions: Train={len(train_ds):,}, Val={len(val_ds):,}, Held-Out Test={len(test_ds):,}")
    print("Strict Policy: Held-out test set is completely isolated and inaccessible during training.")

    # 6. STAGE 1: Head-Only Linear Probe (Backbone Frozen)
    model = init_model
    latest_ckpt_path = output_checkpoint_path.replace("_best.pth", "_latest.pth").replace(".pth", "_latest.pth")

    # Handle resume if requested
    start_stage = 1
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_stage = "head_only"
    training_history = []

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        resume_data = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_data["model_state_dict"])
        best_val_acc = resume_data.get("val_accuracy", 0.0)
        best_val_loss = resume_data.get("val_loss", float("inf"))
        start_epoch = resume_data.get("epoch", 1) + 1
        start_stage = 2 if resume_data.get("stage") == "staged_finetune" else 1

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer_head = optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=1e-2)

    if start_stage == 1:
        print("\n" + "=" * 85)
        print(f"STAGE 1: LINEAR PROBE ON ADAPTED 43-CLASS HEAD ({eff_epochs_head} Epochs)")
        print("=" * 85)

        for ep in range(start_epoch, eff_epochs_head + 1):
            model.train()
            tr_loss_sum = 0.0
            tr_correct = 0
            tr_total = 0

            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer_head.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer_head.step()

                tr_loss_sum += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                tr_correct += (preds == y).sum().item()
                tr_total += y.size(0)

            tr_loss = tr_loss_sum / max(1, tr_total)
            tr_acc = tr_correct / max(1, tr_total)

            val_metrics = evaluate_dataloader(model, val_loader, classes, device)
            v_loss = val_metrics["loss"]
            v_acc = val_metrics["accuracy"]

            training_history.append({
                "stage": "head_only",
                "epoch": ep,
                "train_loss": round(tr_loss, 4),
                "train_accuracy": round(tr_acc, 4),
                "val_loss": v_loss,
                "val_accuracy": v_acc
            })

            # Save latest resumable checkpoint
            torch.save({
                "experiment_name": EXPERIMENT_NAME,
                "epoch": ep,
                "stage": "head_only",
                "val_accuracy": v_acc,
                "val_loss": v_loss,
                "num_classes": num_classes,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "model_state_dict": model.state_dict(),
                "architecture": "mobilenet_v3_small_43class"
            }, latest_ckpt_path)

            is_best = (v_acc > best_val_acc) or (v_acc == best_val_acc and v_loss < best_val_loss)
            tag = ""
            if is_best:
                best_val_acc = v_acc
                best_val_loss = v_loss
                best_epoch = ep
                best_stage = "head_only"
                tag = " [BEST SAVED]"
                torch.save({
                    "experiment_name": EXPERIMENT_NAME,
                    "epoch": ep,
                    "stage": "head_only",
                    "val_accuracy": v_acc,
                    "val_loss": v_loss,
                    "num_classes": num_classes,
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "model_state_dict": model.state_dict(),
                    "architecture": "mobilenet_v3_small_43class"
                }, output_checkpoint_path)

            print(f"Stage 1 [Epoch {ep:02d}/{eff_epochs_head:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2%} | Val Loss: {v_loss:.4f} Acc: {v_acc:.2%}{tag}")

        start_epoch = 1

    # 7. STAGE 2: Staged Fine-Tuning of Upper Layers (Blocks 10-12)
    print("\n" + "=" * 85)
    print(f"STAGE 2: STAGED FINE-TUNING OF TOP BOTTLENECKS ({eff_epochs_finetune} Epochs)")
    print("=" * 85)

    for idx, block in enumerate(model.features):
        if idx >= 10:
            for p in block.parameters():
                p.requires_grad = True

    opt_finetune = optim.AdamW([
        {"params": [p for idx, block in enumerate(model.features) if idx >= 10 for p in block.parameters()], "lr": lr_finetune * 0.1},
        {"params": model.classifier.parameters(), "lr": lr_finetune}
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_finetune, T_max=eff_epochs_finetune, eta_min=1e-6)

    for ep in range(start_epoch, eff_epochs_finetune + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt_finetune.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_finetune.step()

            tr_loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            tr_correct += (preds == y).sum().item()
            tr_total += y.size(0)

        scheduler.step()
        tr_loss = tr_loss_sum / max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        val_metrics = evaluate_dataloader(model, val_loader, classes, device)
        v_loss = val_metrics["loss"]
        v_acc = val_metrics["accuracy"]

        training_history.append({
            "stage": "staged_finetune",
            "epoch": eff_epochs_head + ep,
            "train_loss": round(tr_loss, 4),
            "train_accuracy": round(tr_acc, 4),
            "val_loss": v_loss,
            "val_accuracy": v_acc
        })

        torch.save({
            "experiment_name": EXPERIMENT_NAME,
            "epoch": eff_epochs_head + ep,
            "stage": "staged_finetune",
            "val_accuracy": v_acc,
            "val_loss": v_loss,
            "num_classes": num_classes,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "model_state_dict": model.state_dict(),
            "architecture": "mobilenet_v3_small_43class"
        }, latest_ckpt_path)

        is_best = (v_acc > best_val_acc) or (v_acc == best_val_acc and v_loss < best_val_loss)
        tag = ""
        if is_best:
            best_val_acc = v_acc
            best_val_loss = v_loss
            best_epoch = eff_epochs_head + ep
            best_stage = "staged_finetune"
            tag = " [BEST SAVED]"
            torch.save({
                "experiment_name": EXPERIMENT_NAME,
                "epoch": eff_epochs_head + ep,
                "stage": "staged_finetune",
                "val_accuracy": v_acc,
                "val_loss": v_loss,
                "num_classes": num_classes,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "model_state_dict": model.state_dict(),
                "architecture": "mobilenet_v3_small_43class"
            }, output_checkpoint_path)

        print(f"Stage 2 [Epoch {ep:02d}/{eff_epochs_finetune:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2%} | Val Loss: {v_loss:.4f} Acc: {v_acc:.2%}{tag}")

    # 8. Final Evaluation on Held-Out Test Split (EXACTLY ONCE)
    print("\n" + "=" * 85)
    print("FINAL EVALUATION: HELD-OUT TEST SPLIT (EVALUATED EXACTLY ONCE)")
    print("=" * 85)
    best_ckpt = torch.load(output_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_metrics = evaluate_dataloader(model, test_loader, classes, device)
    elapsed_time = time.time() - start_time
    checkpoint_bytes = os.path.getsize(output_checkpoint_path)

    print(f"Final Test Accuracy:   {test_metrics['accuracy']:.2%} (95% CI: [{test_metrics['accuracy_ci_95'][0]:.2%}, {test_metrics['accuracy_ci_95'][1]:.2%}])")
    print(f"Final Balanced Acc:    {test_metrics['balanced_accuracy']:.2%}")
    print(f"Final Macro-F1:        {test_metrics['macro_f1']:.4f}")
    print(f"Final Weighted-F1:     {test_metrics['weighted_f1']:.4f}")
    print(f"Training Duration:     {elapsed_time:.1f}s")
    print(f"Checkpoint Size:       {checkpoint_bytes:,} bytes ({checkpoint_bytes / (1024*1024):.2f} MB)")
    print("=" * 85)

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "smoke_verification" if smoke_mode else "full_retraining",
        "architecture": "mobilenet_v3_small_43class",
        "num_classes": num_classes,
        "classes": classes,
        "training_config": {
            "epochs_head": eff_epochs_head,
            "epochs_finetune": eff_epochs_finetune,
            "batch_size": batch_size,
            "lr_head": lr_head,
            "lr_finetune": lr_finetune,
            "seed": seed,
            "device": str(device)
        },
        "model_selection": {
            "best_epoch": best_epoch,
            "best_stage": best_stage,
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss
        },
        "held_out_test_metrics": test_metrics,
        "checkpoint": {
            "path": output_checkpoint_path,
            "size_bytes": checkpoint_bytes,
            "size_mb": round(checkpoint_bytes / (1024 * 1024), 2)
        },
        "training_duration_seconds": round(elapsed_time, 2),
        "history": training_history
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Training and evaluation report saved to: {output_report_path}")

    # 9. Package Artifacts ZIP for Download
    zip_path = os.path.join(BASE_DIR, "outputs", "kaggle_indian_43class_clean_v2_artifacts.zip")
    package_artifacts_zip(
        checkpoint_best_path=output_checkpoint_path,
        label_mapping_path=label_mapping_path,
        model_config_path=model_config_path,
        training_report_path=output_report_path,
        manifest_path=manifest_path,
        zip_output_path=zip_path
    )
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[OK] Artifacts ZIP packaged successfully: {zip_path} ({zip_size_mb:.2f} MB)")
    if os.path.exists("/kaggle/working"):
        print(f"     Copied to /kaggle/working/kaggle_indian_43class_clean_v2_artifacts.zip for Kaggle Output download.")

    return report

def main():
    parser = argparse.ArgumentParser(description="Clean Retraining Pipeline for 43-Class Food Recognition")
    parser.add_argument("--manifest", type=str, default="data_manifests/indian_food_43class_repaired_manifest_v2.json")
    parser.add_argument("--checkpoint-out", type=str, default="models/checkpoints/mobilenet_v3_indian_43class_clean_v2_best.pth")
    parser.add_argument("--report-out", type=str, default="outputs/indian_43class_clean_training_report.json")
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="Run in smoke verification mode")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    try:
        run_clean_retraining(
            manifest_path=args.manifest,
            output_checkpoint_path=args.checkpoint_out,
            output_report_path=args.report_out,
            epochs_head=args.epochs_head,
            epochs_finetune=args.epochs_finetune,
            batch_size=args.batch_size,
            smoke_mode=args.smoke,
            resume_path=args.resume
        )
    except DatasetPathValidationError as e:
        print(f"\n[SAFELY STOPPED] {e}")
        sys.exit(0)

if __name__ == "__main__":
    main()
