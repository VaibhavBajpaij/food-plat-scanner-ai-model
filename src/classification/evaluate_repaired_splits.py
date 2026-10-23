"""
Reproducible Evaluation & Split Integrity Verifier for 43-Class Indian Food Classification.
Evaluates model checkpoints against repaired, leakage-free data splits with full metrics:
- Overall Accuracy & Balanced Accuracy
- Macro-F1 & Weighted-F1
- Wilson 95% Confidence Intervals
- Per-class Precision, Recall, F1, and Support
- 43x43 Confusion Matrix
- Checkpoint Contamination / Leakage Detection
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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

def compute_multiclass_metrics(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str]
) -> Dict[str, Any]:
    """
    Computes accuracy, balanced accuracy, macro-F1, weighted-F1,
    per-class precision/recall/F1, and confusion matrix.
    """
    num_classes = len(classes)
    total = len(y_true)
    if total == 0:
        return {"total_samples": 0, "accuracy": 0.0, "macro_f1": 0.0}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / total
    ci_low, ci_high = calculate_wilson_ci(correct, total)

    # 43x43 Confusion matrix
    cm = [[0] * num_classes for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[yt][yp] += 1

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
        "accuracy": round(acc, 4),
        "accuracy_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm
    }

def audit_checkpoint_contamination(
    checkpoint_meta: Dict[str, Any],
    original_manifest_path: str,
    repaired_manifest_path: str
) -> Dict[str, Any]:
    """
    Determines whether images in the repaired test split were present
    in the training set of the existing checkpoint (data contamination / test leakage).
    """
    with open(original_manifest_path, "r", encoding="utf-8") as f:
        orig_man = json.load(f)
    with open(repaired_manifest_path, "r", encoding="utf-8") as f:
        rep_man = json.load(f)

    orig_train_shas = {img["sha256"] for img in orig_man["images"] if img["split"] == "train"}
    rep_test_images = [img for img in rep_man["images"] if img["split"] == "test"]

    contaminated_images = []
    for img in rep_test_images:
        if img["sha256"] in orig_train_shas:
            contaminated_images.append({
                "image_id": img["image_id"],
                "canonical_id": img["canonical_id"],
                "sha256": img["sha256"]
            })

    num_contaminated = len(contaminated_images)
    total_repaired_test = len(rep_test_images)
    contamination_pct = (num_contaminated / max(1, total_repaired_test)) * 100

    can_provide_unbiased_eval = (num_contaminated == 0)

    verdict = {
        "total_repaired_test_images": total_repaired_test,
        "contaminated_test_images_count": num_contaminated,
        "contamination_percentage": round(contamination_pct, 2),
        "can_provide_unbiased_evaluation": can_provide_unbiased_eval,
        "explanation": (
            "CLEAN: Zero repaired test images were in the training set."
            if can_provide_unbiased_eval else
            f"CONTAMINATED: {num_contaminated} of {total_repaired_test} repaired test images "
            f"({contamination_pct:.1f}%) were present in the model's training set! "
            "The existing checkpoint cannot provide an unbiased test evaluation on the repaired splits. "
            "Retraining on the repaired train/val splits is required before final held-out test evaluation."
        )
    }
    return verdict

def build_eval_model(num_classes: int = 43) -> nn.Module:
    """Constructs MobileNetV3-Small architecture for 43 classes."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes)
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate 43-Class Food Model on Repaired Splits")
    parser.add_argument("--checkpoint", type=str, default="audit_artifacts/mobilenet_v3_indian_43class_best.pth")
    parser.add_argument("--manifest", type=str, default="data_manifests/indian_food_43class_repaired_manifest_v2.json")
    parser.add_argument("--original-manifest", type=str, default="audit_artifacts/indian_food_43class_manifest.json")
    args = parser.parse_args()

    print("=" * 85)
    print("43-CLASS INDIAN FOOD MODEL: REPAIRED EVALUATION & CONTAMINATION AUDIT")
    print("=" * 85)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return

    if not os.path.exists(args.manifest):
        print(f"Error: Repaired manifest file not found at {args.manifest}")
        return

    # 1. Inspect Checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Loaded Checkpoint: {args.checkpoint}")
    print(f"  Architecture:    {ckpt.get('architecture', 'Unknown')}")
    print(f"  Epoch:           {ckpt.get('epoch')}")
    print(f"  Stage:           {ckpt.get('stage')}")
    print(f"  Classes in Ckpt: {len(ckpt.get('classes', []))}")

    # 2. Check Contamination
    print("\n--- AUDITING TEST SET CONTAMINATION ---")
    contamination = audit_checkpoint_contamination(
        checkpoint_meta=ckpt,
        original_manifest_path=args.original_manifest,
        repaired_manifest_path=args.manifest
    )
    print(f"Contaminated Test Samples: {contamination['contaminated_test_images_count']} / {contamination['total_repaired_test_images']} ({contamination['contamination_percentage']}%)")
    print(f"Unbiased Evaluation Possible: {contamination['can_provide_unbiased_evaluation']}")
    print(f"Verdict: {contamination['explanation']}")
    print("=" * 85)

if __name__ == "__main__":
    main()
