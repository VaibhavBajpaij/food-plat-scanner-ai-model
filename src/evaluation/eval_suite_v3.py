"""
Phase 3 Comparative Evaluation Suite (V3)
Evaluates 4 Controlled Candidates on the Untouched Held-Out Test Set (25 images) and Validation Set (32 images).

Controlled Candidates:
  - Candidate A: Existing Phase 2 Head-Only Model (mobilenet_v3_food_head_only_v2.pth) - Benchmark Reference
  - Candidate B: Phase 3 Head-Only Model (mobilenet_v3_food_head_only_v3.pth) - Trained on Dataset V3
  - Candidate C: Phase 3 Staged Fine-Tuned Model (mobilenet_v3_food_finetuned_v3.pth) - Bottleneck Blocks 11-12 Unfrozen
  - Candidate D: Phase 3 Simple Linear Probe Diagnostic (mobilenet_v3_food_linear_probe_v3.pth) - Single Linear Classifier

Key Metrics Computed:
  - Overall Accuracy & Macro-F1
  - Per-class Precision, Recall, F1, Support
  - 95% Confidence Intervals (Wilson score interval) for statistical rigor
  - Confusion Matrices
  - Cooked White Rice Specific Recall & Confusion Breakdown
  - Mean Confidence on Correct vs Incorrect Predictions
"""

import os
import sys
import json
import math
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

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
    adjusted_std_error = (z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denominator
    lower = max(0.0, centre_adjusted_probability - adjusted_std_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_std_error)
    return round(lower, 4), round(upper, 4)

def load_candidates(base_dir: str, device: torch.device) -> Dict[str, dict]:
    candidates = {}

    # Candidate A: Existing Phase 2 Head-Only (mobilenet_v3_food_head_only_v2.pth)
    a_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_head_only_v2.pth")
    if os.path.exists(a_path):
        mA = models.mobilenet_v3_small(weights=None)
        in_feat = mA.classifier[3].in_features
        mA.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckptA = torch.load(a_path, map_location=device, weights_only=False)
        mA.load_state_dict(ckptA["model_state_dict"])
        mA = mA.to(device)
        mA.eval()
        candidates["Candidate_A_Phase2_Reference"] = {
            "model": mA,
            "architecture": "MobileNetV3-Small (Frozen Backbone + MLP Head)",
            "checkpoint": "models/checkpoints/mobilenet_v3_food_head_only_v2.pth",
            "description": "Phase 2 Benchmark Model (Trained on Dataset V2: 80 train images)"
        }

    # Candidate B: Phase 3 Head-Only (mobilenet_v3_food_head_only_v3.pth)
    b_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_head_only_v3.pth")
    if os.path.exists(b_path):
        mB = models.mobilenet_v3_small(weights=None)
        in_feat = mB.classifier[3].in_features
        mB.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckptB = torch.load(b_path, map_location=device, weights_only=False)
        mB.load_state_dict(ckptB["model_state_dict"])
        mB = mB.to(device)
        mB.eval()
        candidates["Candidate_B_Head_Only_V3"] = {
            "model": mB,
            "architecture": "MobileNetV3-Small (Frozen Backbone + MLP Head)",
            "checkpoint": "models/checkpoints/mobilenet_v3_food_head_only_v3.pth",
            "description": "Phase 3 Head-Only Model (Trained on Dataset V3: 128 train images with augmented rice)"
        }

    # Candidate C: Phase 3 Staged Fine-Tuned (mobilenet_v3_food_finetuned_v3.pth)
    c_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_finetuned_v3.pth")
    if os.path.exists(c_path):
        mC = models.mobilenet_v3_small(weights=None)
        in_feat = mC.classifier[3].in_features
        mC.classifier[3] = nn.Linear(in_feat, len(FOOD_CLASSES))
        ckptC = torch.load(c_path, map_location=device, weights_only=False)
        mC.load_state_dict(ckptC["model_state_dict"])
        mC = mC.to(device)
        mC.eval()
        candidates["Candidate_C_Staged_FineTuned_V3"] = {
            "model": mC,
            "architecture": "MobileNetV3-Small (Bottleneck Blocks 11-12 & Head Unfrozen)",
            "checkpoint": "models/checkpoints/mobilenet_v3_food_finetuned_v3.pth",
            "description": "Phase 3 Staged Fine-Tuned Model (LR=1e-4, Cosine Decay on Dataset V3)"
        }

    # Candidate D: Phase 3 Simple Linear Probe Diagnostic (mobilenet_v3_food_linear_probe_v3.pth)
    d_path = os.path.join(base_dir, "models", "checkpoints", "mobilenet_v3_food_linear_probe_v3.pth")
    if os.path.exists(d_path):
        mD = models.mobilenet_v3_small(weights=None)
        in_feat = mD.classifier[0].in_features # 576
        mD.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feat, len(FOOD_CLASSES))
        )
        ckptD = torch.load(d_path, map_location=device, weights_only=False)
        mD.load_state_dict(ckptD["model_state_dict"])
        mD = mD.to(device)
        mD.eval()
        candidates["Candidate_D_Linear_Probe_V3"] = {
            "model": mD,
            "architecture": "MobileNetV3-Small (Frozen Backbone + Single Linear Layer)",
            "checkpoint": "models/checkpoints/mobilenet_v3_food_linear_probe_v3.pth",
            "description": "Phase 3 Diagnostic Baseline (Single Linear Layer directly on 576-dim pooling)"
        }

    return candidates

def evaluate_candidate_on_split(model: nn.Module, items: list, base_dir: str, device: torch.device) -> Dict[str, Any]:
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    correct = 0
    total = len(items)
    conf_matrix = {t: {p: 0 for p in FOOD_CLASSES} for t in FOOD_CLASSES}
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_predicted = defaultdict(int)

    correct_confidences = []
    incorrect_confidences = []
    sample_predictions = []

    for item in items:
        true_label = item["canonical_id"]
        img_path = os.path.join(base_dir, item["file_path"])
        img = Image.open(img_path).convert("RGB")
        tensor = eval_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = IDX_TO_CLASS[pred_idx]
        confidence = float(probs[pred_idx])

        is_match = (pred_label == true_label)
        if is_match:
            correct += 1
            per_class_correct[true_label] += 1
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

        per_class_total[true_label] += 1
        per_class_predicted[pred_label] += 1
        conf_matrix[true_label][pred_label] += 1

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [{"class": IDX_TO_CLASS[int(idx)], "prob": round(float(probs[idx]), 4)} for idx in top3_indices]

        sample_predictions.append({
            "image_id": item["image_id"],
            "true_class": true_label,
            "pred_class": pred_label,
            "confidence": round(confidence, 4),
            "correct": is_match,
            "top3": top3,
            "source_url": item.get("source_url", ""),
            "license": item.get("license", "")
        })

    acc = correct / total if total > 0 else 0.0
    acc_ci_low, acc_ci_high = wilson_score_interval(correct, total)

    per_class_metrics = {}
    f1_list = []
    precision_list = []
    recall_list = []

    for c in FOOD_CLASSES:
        tp = per_class_correct[c]
        support = per_class_total[c]
        pred_cnt = per_class_predicted[c]

        prec = tp / pred_cnt if pred_cnt > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        rec_low, rec_high = wilson_score_interval(tp, support)

        per_class_metrics[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "predicted_count": pred_cnt,
            "recall_ci_95": [rec_low, rec_high]
        }
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(precision_list))
    macro_rec = float(np.mean(recall_list))

    return {
        "accuracy": round(acc, 4),
        "accuracy_ci_95": [acc_ci_low, acc_ci_high],
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "correct_count": correct,
        "total_count": total,
        "per_class": per_class_metrics,
        "confusion_matrix": conf_matrix,
        "mean_correct_confidence": round(float(np.mean(correct_confidences)), 4) if correct_confidences else 0.0,
        "mean_incorrect_confidence": round(float(np.mean(incorrect_confidences)), 4) if incorrect_confidences else 0.0,
        "samples": sample_predictions
    }

def run_phase3_evaluation():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manifest_path = os.path.join(base_dir, "data_manifests", "food_dataset_v3_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_items = [r for r in manifest["images"] if r["split"] == "test"]
    val_items = [r for r in manifest["images"] if r["split"] == "val"]

    print("=" * 80)
    print("PHASE 3 CONTROLLED EXPERIMENT EVALUATION SUITE")
    print(f"Held-Out Test Set (Untouched): {len(test_items)} images")
    print(f"Validation Set:               {len(val_items)} images")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = load_candidates(base_dir, device)
    print(f"Loaded {len(candidates)} candidates: {list(candidates.keys())}")

    results = {}
    for c_id, c_data in candidates.items():
        print(f"\nEvaluating: {c_id} ({c_data['description']})...")
        m = c_data["model"]
        test_res = evaluate_candidate_on_split(m, test_items, base_dir, device)
        val_res = evaluate_candidate_on_split(m, val_items, base_dir, device)

        results[c_id] = {
            "checkpoint": c_data["checkpoint"],
            "architecture": c_data["architecture"],
            "description": c_data["description"],
            "held_out_test": test_res,
            "validation": val_res
        }

        # Print quick summary
        t = test_res
        rice_m = t["per_class"]["rice_white_cooked"]
        print(f"  Test Accuracy: {t['accuracy']:.2%} (95% CI: [{t['accuracy_ci_95'][0]:.1%}, {t['accuracy_ci_95'][1]:.1%}]) | Macro-F1: {t['macro_f1']:.4f}")
        print(f"  Rice Recall:   {rice_m['recall']:.2%} ({rice_m['precision']:.2%} prec, F1: {rice_m['f1_score']:.4f}) [CI: {rice_m['recall_ci_95'][0]:.1%}-{rice_m['recall_ci_95'][1]:.1%}]")

    out_path = os.path.join(base_dir, "outputs", "phase3_evaluation_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_protocol": "Held-Out Test Split (25 untouched images)",
            "results": results
        }, f, indent=2)

    print(f"\nPhase 3 Evaluation Report saved to: {out_path}")
    return results

if __name__ == "__main__":
    run_phase3_evaluation()
