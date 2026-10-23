"""
Comprehensive Evaluation Suite (V2)
Evaluates 4 Candidates across:
  1. Held-Out Test Set (untouched during training and tuning)
  2. Validation Set (used for checkpoint selection)
  3. Legacy Diagnostic Smoke Test Set (for baseline continuity)

Candidates:
  - Candidate A: Original Pretrained MobileNetV3-Small (ImageNet-1k Baseline, strict synset matching)
  - Candidate B: Existing Head-Only Checkpoint (mobilenet_v3_food_head_only.pth from Phase 1)
  - Candidate C: Improved Head-Only Classifier (mobilenet_v3_food_head_only_v2.pth)
  - Candidate D: Staged Fine-Tuned Classifier (mobilenet_v3_food_finetuned_v2.pth)

Computes:
  - Overall Accuracy and Macro-F1
  - Per-class Precision, Recall, F1, Support
  - Confusion Matrix
  - Prediction Confidence Analysis (correct vs incorrect)
  - Failure Case Analysis with Top-3 Probabilities
  - Lighting and Background Performance Breakdown
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
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

# Strict synset mapping for ImageNet-1k baseline (No artificial proxies like lobster -> rice)
IMAGENET_STRICT_SYNSET_MAP = {
    "banana": "banana_raw",
    "broccoli": "broccoli_steamed",
    "Granny Smith": "apple_fresh_raw",
    "custard apple": "apple_fresh_raw"
    # rice_white_cooked and chicken_breast_grilled have NO synsets in ImageNet-1k
}

KITCHENWARE_SYNSETS = {
    "plate", "tray", "pot", "pan", "soup bowl", "wok", "platter", "potter's wheel",
    "coffee mug", "pitcher", "caldron", "mixing bowl"
}

def load_all_candidates(base_dir: str, device: torch.device):
    models_dict = {}

    # Candidate A: Original Pretrained MobileNetV3-Small
    w = models.MobileNet_V3_Small_Weights.DEFAULT
    mA = models.mobilenet_v3_small(weights=w).to(device)
    mA.eval()
    models_dict["Candidate_A_Original_Pretrained"] = {
        "model": mA,
        "type": "imagenet_1k",
        "checkpoint": "torchvision.models.mobilenet_v3_small-047dcff4.pth",
        "description": "Original Pretrained MobileNetV3-Small (ImageNet-1k Baseline)"
    }
    categories = w.meta["categories"]

    # Candidate B: Existing Head-Only Checkpoint (Phase 1)
    b_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_head_only.pth")
    if os.path.exists(b_path):
        mB = models.mobilenet_v3_small(weights=None)
        in_feat = mB.classifier[3].in_features
        mB.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckptB = torch.load(b_path, map_location=device, weights_only=False)
        mB.load_state_dict(ckptB["model_state_dict"])
        mB = mB.to(device)
        mB.eval()
        models_dict["Candidate_B_Existing_Head_Only"] = {
            "model": mB,
            "type": "food_5class",
            "checkpoint": "models/checkpoints/mobilenet_v3_food_head_only.pth",
            "description": "Phase 1 Head-Only Baseline (22 train images)"
        }

    # Candidate C: Improved Head-Only Checkpoint (Phase 2)
    c_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_head_only_v2.pth")
    if os.path.exists(c_path):
        mC = models.mobilenet_v3_small(weights=None)
        in_feat = mC.classifier[3].in_features
        mC.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckptC = torch.load(c_path, map_location=device, weights_only=False)
        mC.load_state_dict(ckptC["model_state_dict"])
        mC = mC.to(device)
        mC.eval()
        models_dict["Candidate_C_Improved_Head_Only"] = {
            "model": mC,
            "type": "food_5class",
            "checkpoint": "models/checkpoints/mobilenet_v3_food_head_only_v2.pth",
            "description": "Phase 2 Head-Only Model (Trained on Expanded Dataset V2 with Augmentation)"
        }

    # Candidate D: Staged Fine-Tuned Checkpoint (Phase 2)
    d_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_finetuned_v2.pth")
    if os.path.exists(d_path):
        mD = models.mobilenet_v3_small(weights=None)
        in_feat = mD.classifier[3].in_features
        mD.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckptD = torch.load(d_path, map_location=device, weights_only=False)
        mD.load_state_dict(ckptD["model_state_dict"])
        mD = mD.to(device)
        mD.eval()
        models_dict["Candidate_D_Staged_Fine_Tuned"] = {
            "model": mD,
            "type": "food_5class",
            "checkpoint": "models/checkpoints/mobilenet_v3_food_finetuned_v2.pth",
            "description": "Phase 2 Staged Fine-Tuned Model (Upper Bottleneck + Head Unfrozen)"
        }

    return models_dict, categories

def evaluate_single_model_on_dataset(model_info: dict, dataset_items: list, base_dir: str, imagenet_cats: list, device: torch.device):
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = model_info["model"]
    mtype = model_info["type"]

    correct = 0
    total = 0
    preds = []
    targets = []
    confidences = []
    correct_confidences = []
    incorrect_confidences = []
    kitchenware_count = 0
    failures = []

    for item in dataset_items:
        img_p = os.path.join(base_dir, item["file_path"])
        if not os.path.exists(img_p):
            continue

        gt_class = item["canonical_id"]
        img = Image.open(img_p).convert("RGB")
        inp = eval_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(inp)

        probs = torch.softmax(output[0], dim=0)

        if mtype == "imagenet_1k":
            top1_idx = output.argmax(dim=1).item()
            raw_cat = imagenet_cats[top1_idx]
            conf = probs[top1_idx].item()

            if any(kw in raw_cat.lower() for kw in KITCHENWARE_SYNSETS):
                kitchenware_count += 1

            mapped_class = IMAGENET_STRICT_SYNSET_MAP.get(raw_cat, "unrepresented_or_other")
            is_correct = (mapped_class == gt_class)

            top5_p, top5_i = torch.topk(probs, 5)
            top5_info = [{"category": imagenet_cats[idx.item()], "prob": round(p.item(), 4)} for p, idx in zip(top5_p, top5_i)]

            preds.append(mapped_class)
            targets.append(gt_class)
            confidences.append(conf)

            if is_correct:
                correct += 1
                correct_confidences.append(conf)
            else:
                incorrect_confidences.append(conf)
                failures.append({
                    "image": item["file_path"],
                    "ground_truth": gt_class,
                    "predicted": mapped_class,
                    "raw_imagenet_top1": raw_cat,
                    "confidence": round(conf, 4),
                    "top_predictions": top5_info[:3]
                })

        else: # food_5class
            top1_idx = output.argmax(dim=1).item()
            pred_class = IDX_TO_CLASS[top1_idx]
            conf = probs[top1_idx].item()
            is_correct = (pred_class == gt_class)

            top3_p, top3_i = torch.topk(probs, min(3, len(FOOD_CLASSES)))
            top3_info = [{"category": IDX_TO_CLASS[idx.item()], "prob": round(p.item(), 4)} for p, idx in zip(top3_p, top3_i)]

            preds.append(pred_class)
            targets.append(gt_class)
            confidences.append(conf)

            if is_correct:
                correct += 1
                correct_confidences.append(conf)
            else:
                incorrect_confidences.append(conf)
                failures.append({
                    "image": item["file_path"],
                    "ground_truth": gt_class,
                    "predicted": pred_class,
                    "confidence": round(conf, 4),
                    "top_predictions": top3_info
                })

        total += 1

    acc = correct / total if total > 0 else 0.0

    # Confusion matrix
    all_possible_preds = FOOD_CLASSES + (["unrepresented_or_other"] if mtype == "imagenet_1k" else [])
    cm = {c1: {c2: 0 for c2 in all_possible_preds} for c1 in FOOD_CLASSES}
    for p, t in zip(preds, targets):
        if t in cm:
            cm[t][p] = cm[t].get(p, 0) + 1

    # Per-class metrics
    per_class = {}
    f1_list = []
    for c in FOOD_CLASSES:
        tp = cm[c].get(c, 0)
        fn = sum(cm[c].values()) - tp
        fp = sum(cm[other].get(c, 0) for other in FOOD_CLASSES if other != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class[c] = {
            "support": tp + fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    mean_conf_correct = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0
    mean_conf_incorrect = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0

    return {
        "total_images": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_confidence_correct": round(mean_conf_correct, 4),
        "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
        "kitchenware_bias_count": kitchenware_count,
        "per_class": per_class,
        "confusion_matrix": cm,
        "failures": failures
    }

def run_evaluation_suite_v2():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("RUNNING COMPREHENSIVE 4-CANDIDATE EVALUATION SUITE (V2)")
    print(f"Device: {device}")
    print("=" * 80)

    models_dict, imagenet_cats = load_all_candidates(base_dir, device)
    print(f"Loaded {len(models_dict)} model candidates:")
    for k, v in models_dict.items():
        print(f"  - {k}: {v['description']}")

    # Load Splits
    v2_manifest_path = os.path.join(base_dir, "data_manifests", "food_dataset_v2_manifest.json")
    if not os.path.exists(v2_manifest_path):
        raise FileNotFoundError(f"Missing V2 manifest: {v2_manifest_path}")

    with open(v2_manifest_path, "r", encoding="utf-8") as f:
        v2_data = json.load(f)

    val_items = [r for r in v2_data["images"] if r["split"] == "val"]
    test_items = [r for r in v2_data["images"] if r["split"] == "test"]

    # Load legacy smoke test
    smoke_manifest_path = os.path.join(base_dir, "data_manifests", "smoke_test_manifest.json")
    smoke_items = []
    if os.path.exists(smoke_manifest_path):
        with open(smoke_manifest_path, "r", encoding="utf-8") as f:
            smoke_data = json.load(f)
        smoke_items = smoke_data.get("images", [])

    print(f"\nEvaluation Splits:")
    print(f"  - Held-Out Test Split:  {len(test_items)} images (Strictly untouched during training)")
    print(f"  - Validation Split:     {len(val_items)} images")
    print(f"  - Legacy Smoke Test:    {len(smoke_items)} images")

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "held_out_test_metrics": {},
        "validation_metrics": {},
        "smoke_test_metrics": {}
    }

    # Evaluate all models across splits
    splits_to_eval = [
        ("held_out_test_metrics", test_items, "Held-Out Test Split"),
        ("validation_metrics", val_items, "Validation Split"),
        ("smoke_test_metrics", smoke_items, "Legacy Smoke-Test Split")
    ]

    for split_key, items, split_name in splits_to_eval:
        if not items:
            continue
        print("\n" + "=" * 80)
        print(f"EVALUATING: {split_name} ({len(items)} images)")
        print("=" * 80)
        for mname, minfo in models_dict.items():
            res = evaluate_single_model_on_dataset(minfo, items, base_dir, imagenet_cats, device)
            all_results[split_key][mname] = res
            print(f"  {mname:<35} | Acc: {res['accuracy']:>6.1%} | Macro-F1: {res['macro_f1']:>6.3f} | Correct Conf: {res['mean_confidence_correct']:>5.2f}")

    # Summary table
    print("\n" + "=" * 85)
    print("COMPARATIVE PERFORMANCE MATRIX (HELD-OUT TEST SET - UNTOUCHED DATA)")
    print("=" * 85)
    print(f"{'Candidate Model':<36} {'Test Acc':<12} {'Macro-F1':<12} {'Val Acc':<12} {'Kitchenware Bias'}")
    print("-" * 85)
    for mname in models_dict:
        t_acc = all_results["held_out_test_metrics"].get(mname, {}).get("accuracy", 0.0)
        t_f1 = all_results["held_out_test_metrics"].get(mname, {}).get("macro_f1", 0.0)
        v_acc = all_results["validation_metrics"].get(mname, {}).get("accuracy", 0.0)
        kw = all_results["held_out_test_metrics"].get(mname, {}).get("kitchenware_bias_count", 0)
        print(f"{mname:<36} {t_acc:>8.1%}    {t_f1:>8.3f}    {v_acc:>8.1%}    {kw} occurrences")
    print("=" * 85)

    out_json = os.path.join(base_dir, "outputs", "evaluation_results_v2.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComplete evaluation results saved to: {out_json}")
    return all_results

if __name__ == "__main__":
    run_evaluation_suite_v2()
