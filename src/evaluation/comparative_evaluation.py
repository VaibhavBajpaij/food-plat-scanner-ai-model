"""
Comparative Evaluation Suite
Benchmarks:
  1. Original Pretrained MobileNetV3-Small (ImageNet Baseline)
  2. Experiment 1: Head-Only Fine-Tuned Model
  3. Experiment 2: Staged Fine-Tuned Model (Upper Layers Unfrozen)
Evaluated across:
  - Independent Validation Split (finetune/val)
  - Independent Smoke-Test Set (eval/smoke_test)
Outputs per-class precision, recall, confusion matrices, and failure examples.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
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

IMAGENET_TO_FOOD_MAP = {
    "banana": "banana_raw",
    "orange": "apple_fresh_raw",        # Nearest round fruit proxy in ImageNet
    "Granny Smith": "apple_fresh_raw",
    "custard apple": "apple_fresh_raw",
    "fig": "apple_fresh_raw",
    "broccoli": "broccoli_steamed",
    "meat loaf": "chicken_breast_grilled",
    "rotisserie": "chicken_breast_grilled",
    "American lobster": "rice_white_cooked",
    "carbonara": "rice_white_cooked"
}

def load_models(base_dir: str, device: torch.device):
    # Model 0: Pretrained ImageNet
    w = models.MobileNet_V3_Small_Weights.DEFAULT
    m0 = models.mobilenet_v3_small(weights=w).to(device)
    m0.eval()

    # Model 1: Head Only
    m1_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_head_only.pth")
    m1 = None
    if os.path.exists(m1_path):
        m1 = models.mobilenet_v3_small(weights=None)
        in_feat = m1.classifier[3].in_features
        m1.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckpt1 = torch.load(m1_path, map_location=device)
        m1.load_state_dict(ckpt1["model_state_dict"])
        m1 = m1.to(device)
        m1.eval()

    # Model 2: Staged Fine-Tuned
    m2_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_finetuned.pth")
    m2 = None
    if os.path.exists(m2_path):
        m2 = models.mobilenet_v3_small(weights=None)
        in_feat = m2.classifier[3].in_features
        m2.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckpt2 = torch.load(m2_path, map_location=device)
        m2.load_state_dict(ckpt2["model_state_dict"])
        m2 = m2.to(device)
        m2.eval()

    return m0, m1, m2, w.meta["categories"]

def evaluate_dataset(dataset_items: list, base_dir: str, m0, m1, m2, imagenet_cats, device: torch.device):
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {
        "m0_original": {"correct": 0, "total": 0, "preds": [], "targets": [], "failures": []},
        "m1_head_only": {"correct": 0, "total": 0, "preds": [], "targets": [], "failures": []},
        "m2_finetuned": {"correct": 0, "total": 0, "preds": [], "targets": [], "failures": []}
    }

    for item in dataset_items:
        img_p = os.path.join(base_dir, item["file_path"])
        if not os.path.exists(img_p):
            continue
        gt_class = item["canonical_id"]
        gt_idx = CLASS_TO_IDX[gt_class]

        img = Image.open(img_p).convert("RGB")
        inp = eval_tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out0 = m0(inp)
            top1_cat = imagenet_cats[out0.argmax().item()]
            m0_mapped = IMAGENET_TO_FOOD_MAP.get(top1_cat, "unknown_other")
            is_m0_correct = (m0_mapped == gt_class)
            results["m0_original"]["total"] += 1
            if is_m0_correct:
                results["m0_original"]["correct"] += 1
            else:
                results["m0_original"]["failures"].append({
                    "image": item["file_path"],
                    "ground_truth": gt_class,
                    "predicted_raw_imagenet": top1_cat,
                    "mapped_prediction": m0_mapped
                })
            results["m0_original"]["preds"].append(m0_mapped)
            results["m0_original"]["targets"].append(gt_class)

            # Model 1
            if m1:
                out1 = m1(inp)
                pred1_idx = out1.argmax().item()
                pred1_class = IDX_TO_CLASS[pred1_idx]
                results["m1_head_only"]["total"] += 1
                if pred1_class == gt_class:
                    results["m1_head_only"]["correct"] += 1
                else:
                    results["m1_head_only"]["failures"].append({
                        "image": item["file_path"],
                        "ground_truth": gt_class,
                        "predicted": pred1_class
                    })
                results["m1_head_only"]["preds"].append(pred1_class)
                results["m1_head_only"]["targets"].append(gt_class)

            # Model 2
            if m2:
                out2 = m2(inp)
                pred2_idx = out2.argmax().item()
                pred2_class = IDX_TO_CLASS[pred2_idx]
                results["m2_finetuned"]["total"] += 1
                if pred2_class == gt_class:
                    results["m2_finetuned"]["correct"] += 1
                else:
                    results["m2_finetuned"]["failures"].append({
                        "image": item["file_path"],
                        "ground_truth": gt_class,
                        "predicted": pred2_class
                    })
                results["m2_finetuned"]["preds"].append(pred2_class)
                results["m2_finetuned"]["targets"].append(gt_class)

    return results

def compute_metrics(eval_res: dict) -> dict:
    summary = {}
    for mname, data in eval_res.items():
        if data["total"] == 0:
            continue
        acc = data["correct"] / data["total"]
        preds = data["preds"]
        targets = data["targets"]

        # Confusion Matrix
        cm = {c1: {c2: 0 for c2 in FOOD_CLASSES + ["unknown_other"]} for c1 in FOOD_CLASSES}
        for p, t in zip(preds, targets):
            if t in cm:
                cm[t][p] = cm[t].get(p, 0) + 1

        # Per-class precision, recall
        per_class = {}
        for c in FOOD_CLASSES:
            tp = cm[c].get(c, 0)
            fn = sum(cm[c].values()) - tp
            fp = sum(cm[other].get(c, 0) for other in FOOD_CLASSES if other != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[c] = {
                "support": tp + fn,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3)
            }

        summary[mname] = {
            "total_images": data["total"],
            "accuracy": round(acc, 4),
            "per_class": per_class,
            "confusion_matrix": cm,
            "failure_samples": data["failures"][:5]
        }
    return summary

def run_comparative_benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("RUNNING THREE-WAY COMPARATIVE BENCHMARK")
    print(f"Device: {device}")
    print("=" * 70)

    m0, m1, m2, imagenet_cats = load_models(base_dir, device)

    # 1. Independent Validation Set
    finetune_manifest_p = os.path.join(base_dir, "data_manifests", "finetune_dataset_manifest.json")
    with open(finetune_manifest_p, "r", encoding="utf-8") as f:
        ft_manifest = json.load(f)
    val_items = [r for r in ft_manifest["images"] if r["split"] == "val"]

    # 2. Independent Smoke Test Set
    smoke_manifest_p = os.path.join(base_dir, "data_manifests", "smoke_test_manifest.json")
    with open(smoke_manifest_p, "r", encoding="utf-8") as f:
        smoke_manifest = json.load(f)
    smoke_items = smoke_manifest.get("images", [])

    print(f"\nEvaluating on Validation Split ({len(val_items)} images)...")
    val_raw = evaluate_dataset(val_items, base_dir, m0, m1, m2, imagenet_cats, device)
    val_metrics = compute_metrics(val_raw)

    print(f"Evaluating on Smoke Test Set ({len(smoke_items)} images)...")
    smoke_raw = evaluate_dataset(smoke_items, base_dir, m0, m1, m2, imagenet_cats, device)
    smoke_metrics = compute_metrics(smoke_raw)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "validation_set_metrics": val_metrics,
        "smoke_test_metrics": smoke_metrics
    }

    out_p = os.path.join(base_dir, "outputs", "comparative_evaluation_results.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Val Acc (10 imgs)':<18} {'Smoke Test Acc (17 imgs)':<25}")
    print("-" * 70)
    for k, name in [("m0_original", "0. Original ImageNet"),
                    ("m1_head_only", "1. Head-Only Trained"),
                    ("m2_finetuned", "2. Staged Fine-Tuned")]:
        v_acc = val_metrics.get(k, {}).get("accuracy", "N/A")
        s_acc = smoke_metrics.get(k, {}).get("accuracy", "N/A")
        v_str = f"{v_acc:.1%}" if isinstance(v_acc, float) else str(v_acc)
        s_str = f"{s_acc:.1%}" if isinstance(s_acc, float) else str(s_acc)
        print(f"{name:<30} {v_str:<18} {s_str:<25}")
    print("=" * 70)
    print(f"Results saved to: {out_p}")
    return report

if __name__ == "__main__":
    run_comparative_benchmark()
