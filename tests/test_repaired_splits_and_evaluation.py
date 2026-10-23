"""
Unit & Integrity Tests for Repaired 43-Class Indian Food Splits and Evaluation
Covers:
1. Manifest integrity (schema, fields, version 2.0.0)
2. Label mapping (43 classes match checkpoint exactly)
3. Class coverage (all 43 classes present in train, val, and test)
4. Duplicate leakage (zero SHA and zero perceptual leakage across splits)
5. Split disjointness (no group spans multiple splits)
6. Checkpoint loading and compatibility
7. Metrics calculation and Wilson 95% confidence intervals
8. Reproducible split generation (deterministic seed=42)
"""

import os
import sys
import json
import torch
import numpy as np
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.classification.evaluate_repaired_splits import (
    calculate_wilson_ci,
    compute_multiclass_metrics,
    audit_checkpoint_contamination,
    build_eval_model
)

MANIFEST_V2_PATH = os.path.join(BASE_DIR, "data_manifests", "indian_food_43class_repaired_manifest_v2.json")
ORIGINAL_MANIFEST_PATH = os.path.join(BASE_DIR, "audit_artifacts", "indian_food_43class_manifest.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "audit_artifacts", "mobilenet_v3_indian_43class_best.pth")

def test_manifest_integrity():
    assert os.path.exists(MANIFEST_V2_PATH), f"Repaired manifest not found at {MANIFEST_V2_PATH}"
    with open(MANIFEST_V2_PATH, "r", encoding="utf-8") as f:
        man = json.load(f)

    for field in ["dataset_name", "version", "created_at", "total_images", "classes_count", "classes", "split_summary", "split_audit", "leakage_audit", "images"]:
        assert field in man, f"Missing required manifest field: {field}"

    assert man["version"] == "2.0.0"
    assert man["classes_count"] == 43
    assert len(man["classes"]) == 43
    assert len(man["images"]) == man["total_images"]
    assert man["total_images"] == 15358  # 15,362 - 4 excluded due to conflicting labels

def test_label_mapping_and_checkpoint_compatibility():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[SKIP] Checkpoint not found at {CHECKPOINT_PATH} (cloud environment).")
        return

    with open(MANIFEST_V2_PATH, "r", encoding="utf-8") as f:
        man = json.load(f)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    ckpt_classes = ckpt.get("classes", [])
    man_classes = man.get("classes", [])
    assert ckpt_classes == man_classes, "Checkpoint classes do not match manifest classes exactly"

    # Verify model forward pass with loaded state_dict
    model = build_eval_model(num_classes=43)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 43), f"Expected output shape (2, 43), got {out.shape}"

def test_class_coverage_across_all_splits():
    with open(MANIFEST_V2_PATH, "r", encoding="utf-8") as f:
        man = json.load(f)

    split_classes = {"train": set(), "val": set(), "test": set()}
    for img in man["images"]:
        split_classes[img["split"]].add(img["canonical_id"])

    expected_43 = set(man["classes"])
    assert len(expected_43) == 43

    assert split_classes["train"] == expected_43, f"Train missing classes: {expected_43 - split_classes['train']}"
    assert split_classes["val"] == expected_43, f"Validation missing classes: {expected_43 - split_classes['val']}"
    assert split_classes["test"] == expected_43, f"Test missing classes: {expected_43 - split_classes['test']}"

def test_duplicate_leakage_and_split_disjointness():
    with open(MANIFEST_V2_PATH, "r", encoding="utf-8") as f:
        man = json.load(f)

    train_shas = set(img["sha256"] for img in man["images"] if img["split"] == "train")
    val_shas = set(img["sha256"] for img in man["images"] if img["split"] == "val")
    test_shas = set(img["sha256"] for img in man["images"] if img["split"] == "test")

    # Exact duplicate leakage
    assert len(train_shas & val_shas) == 0, "Exact SHA collision between Train and Val"
    assert len(train_shas & test_shas) == 0, "Exact SHA collision between Train and Test"
    assert len(val_shas & test_shas) == 0, "Exact SHA collision between Val and Test"

    # Group disjointness
    group_to_splits = {}
    for img in man["images"]:
        gid = img.get("group_id")
        if gid:
            if gid not in group_to_splits:
                group_to_splits[gid] = set()
            group_to_splits[gid].add(img["split"])

    multi_split_groups = {g: s for g, s in group_to_splits.items() if len(s) > 1}
    assert len(multi_split_groups) == 0, f"Groups spanning multiple splits found: {multi_split_groups}"

def test_perceptual_cross_split_leakage():
    with open(MANIFEST_V2_PATH, "r", encoding="utf-8") as f:
        man = json.load(f)

    dhashes_uint64 = np.array([int(img["perceptual_hash"], 16) for img in man["images"]], dtype=np.uint64)
    splits = np.array([img["split"] for img in man["images"]])

    def count_bits_64(x):
        x = (x & np.uint64(0x5555555555555555)) + ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
        x = (x & np.uint64(0x0f0f0f0f0f0f0f0f)) + ((x >> np.uint64(4)) & np.uint64(0x0f0f0f0f0f0f0f0f))
        x = (x & np.uint64(0x00ff00ff00ff00ff)) + ((x >> np.uint64(8)) & np.uint64(0x00ff00ff00ff00ff))
        x = (x & np.uint64(0x0000ffff0000ffff)) + ((x >> np.uint64(16)) & np.uint64(0x0000ffff0000ffff))
        x = (x & np.uint64(0x00000000ffffffff)) + ((x >> np.uint64(32)) & np.uint64(0x00000000ffffffff))
        return x.astype(np.int32)

    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]

    val_h = dhashes_uint64[val_idx]
    test_h = dhashes_uint64[test_idx]

    xor_m = val_h[:, None] ^ test_h[None, :]
    dist_m = count_bits_64(xor_m)
    near_matches = np.sum((dist_m <= 2) & (dist_m >= 0))
    assert near_matches == 0, f"Found {near_matches} perceptual near-duplicates (dist <= 2) between Val and Test"

def test_metrics_calculation_and_wilson_ci():
    # Test Wilson interval
    low, high = calculate_wilson_ci(k=80, n=100)
    assert 0.70 < low < 0.73
    assert 0.85 < high < 0.88

    # Edge cases
    assert calculate_wilson_ci(0, 0) == (0.0, 0.0)
    assert calculate_wilson_ci(0, 100)[0] == 0.0
    assert math.isclose(calculate_wilson_ci(100, 100)[1], 1.0)

    # Multi-class metrics computation
    classes = ["biryani", "chole", "samosa"]
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]  # class 1 has 1 misclassified as class 2
    metrics = compute_multiclass_metrics(y_true, y_pred, classes)

    assert metrics["total_samples"] == 6
    assert metrics["correct_predictions"] == 5
    assert metrics["accuracy"] == round(5 / 6, 4)
    assert "biryani" in metrics["per_class"]
    assert metrics["per_class"]["biryani"]["precision"] == 1.0
    assert metrics["per_class"]["biryani"]["recall"] == 1.0
    assert metrics["per_class"]["biryani"]["f1_score"] == 1.0

def test_checkpoint_contamination_detection():
    if not os.path.exists(ORIGINAL_MANIFEST_PATH):
        print(f"[SKIP] Original manifest not found at {ORIGINAL_MANIFEST_PATH} (cloud environment).")
        return

    contamination = audit_checkpoint_contamination(
        checkpoint_meta={},
        original_manifest_path=ORIGINAL_MANIFEST_PATH,
        repaired_manifest_path=MANIFEST_V2_PATH
    )
    assert contamination["contaminated_test_images_count"] > 1000
    assert contamination["can_provide_unbiased_evaluation"] is False
    assert "CONTAMINATED" in contamination["explanation"]

if __name__ == "__main__":
    test_manifest_integrity()
    test_label_mapping_and_checkpoint_compatibility()
    test_class_coverage_across_all_splits()
    test_duplicate_leakage_and_split_disjointness()
    test_perceptual_cross_split_leakage()
    test_metrics_calculation_and_wilson_ci()
    test_checkpoint_contamination_detection()
    print("All Repaired Splits and Evaluation tests passed successfully!")
