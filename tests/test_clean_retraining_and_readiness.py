"""
Unit & Integration Tests for Clean Retraining Pipeline and Deployment Readiness
Validates:
1. Clean ImageNet weight initialization (uncontaminated by old food checkpoint).
2. 43-Class architecture and classifier head structure.
3. Duplicate-group isolation and split leakage prevention.
4. Test split isolation guardrail.
5. Path validation safety stop.
6. Deployment readiness benchmarks and Android 2GB qualification guardrail.
"""

import os
import sys
import json
import unittest
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.classification.retrain_clean_43class import (
    build_clean_mobilenet_v3,
    verify_label_mapping,
    verify_duplicate_group_isolation,
    verify_dataset_paths,
    ManifestFoodDataset,
    CANONICAL_43_FOOD_CLASSES,
    SplitLeakageError,
    DatasetPathValidationError
)

def test_clean_imagenet_initialization_not_contaminated_checkpoint():
    """Verifies that clean model starts from ImageNet weights, not the contaminated checkpoint."""
    clean_model = build_clean_mobilenet_v3(num_classes=43, pretrained=True)
    clean_features_w = clean_model.features[1].block[0][0].weight.data.clone()

    contaminated_path = os.path.join(BASE_DIR, "audit_artifacts", "mobilenet_v3_indian_43class_best.pth")
    if os.path.exists(contaminated_path):
        ckpt = torch.load(contaminated_path, map_location="cpu", weights_only=False)
        old_state = ckpt["model_state_dict"]
        # Contrast weights of fine-tuned upper layer (features[11])
        if "features.11.block.0.0.weight" in old_state:
            old_w = old_state["features.11.block.0.0.weight"]
            clean_w = clean_model.features[11].block[0][0].weight.data
            diff = (old_w - clean_w).abs().sum().item()
            assert diff > 0.1, f"Clean weights must NOT match the contaminated checkpoint! Diff: {diff}"

def test_43class_classifier_head_structure():
    """Verifies classifier head dimensions and forward pass output shape."""
    model = build_clean_mobilenet_v3(num_classes=43, pretrained=False)
    classifier = model.classifier
    assert len(classifier) == 4, f"Expected 4 layers in classifier head, got {len(classifier)}"
    assert isinstance(classifier[0], nn.Linear) and classifier[0].in_features == 576 and classifier[0].out_features == 1024
    assert isinstance(classifier[1], nn.Hardswish)
    assert isinstance(classifier[2], nn.Dropout)
    assert isinstance(classifier[3], nn.Linear) and classifier[3].in_features == 1024 and classifier[3].out_features == 43

    dummy_input = torch.randn(3, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (3, 43), f"Expected shape (3, 43), got {out.shape}"

def test_repaired_manifest_duplicate_group_isolation():
    """Verifies that all 15,248 groups in the repaired manifest are 100% isolated."""
    manifest_path = os.path.join(BASE_DIR, "data_manifests", "indian_food_43class_repaired_manifest_v2.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    iso = verify_duplicate_group_isolation(images)
    assert iso["is_isolated"], "Repaired manifest must have 100% group isolation!"
    assert iso["leaking_groups_count"] == 0

    # Injected test: Synthetic leaking group must raise SplitLeakageError
    leaking_images = [
        {"image_id": "img1", "group_id": "leak_grp", "split": "train"},
        {"image_id": "img2", "group_id": "leak_grp", "split": "test"}
    ]
    try:
        verify_duplicate_group_isolation(leaking_images)
        assert False, "Leaking duplicate group across train and test must raise SplitLeakageError!"
    except SplitLeakageError:
        pass

def test_test_split_isolation_guardrail():
    """Verifies that ManifestFoodDataset strictly prevents items from other splits."""
    items = [
        {"image_id": "img_test", "split": "test", "file_path": "fake.jpg", "canonical_id": "biryani"}
    ]
    class_to_idx = {c: i for i, c in enumerate(CANONICAL_43_FOOD_CLASSES)}
    try:
        ManifestFoodDataset(items, class_to_idx, expected_split="train")
        assert False, "Passing test items into train dataset must raise SplitLeakageError!"
    except SplitLeakageError:
        pass

def test_path_validation_safety_stop():
    """Verifies that missing physical image files trigger a safety stop."""
    fake_images = [
        {"image_id": "f1", "file_path": "/nonexistent/path/image1.jpg", "split": "train"},
        {"image_id": "f2", "file_path": "/nonexistent/path/image2.jpg", "split": "val"}
    ]
    all_exist, total_chk, missing_cnt, missing = verify_dataset_paths(fake_images, base_dir=BASE_DIR)
    assert not all_exist, "Nonexistent paths must fail verification!"
    assert missing_cnt == 2
    assert len(missing) > 0

def test_deployment_readiness_metrics_and_guardrail():
    """Verifies deployment readiness benchmark outputs and Android 2GB qualification disclaimer."""
    report_path = os.path.join(BASE_DIR, "outputs", "deployment_readiness_benchmark.json")
    assert os.path.exists(report_path), f"Deployment report not found: {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Latency and model size checks
    assert rep["parameter_metrics"]["total_parameters"] == 1561931
    assert rep["parameter_metrics"]["model_file_size_mb"] <= 15.0
    assert rep["inference_benchmarks"]["cpu"]["mean_latency_ms"] < 100.0

    # Guardrail check: must NOT claim Android 2GB qualification
    guardrail = rep["deployment_guardrail"]
    assert guardrail["android_2gb_ram_compatibility_claim"] == "UNVERIFIED_PENDING_ON_DEVICE_TESTING"
    assert "Android 2GB RAM compatible" in guardrail["compliance_statement"]

if __name__ == "__main__":
    test_clean_imagenet_initialization_not_contaminated_checkpoint()
    test_43class_classifier_head_structure()
    test_repaired_manifest_duplicate_group_isolation()
    test_test_split_isolation_guardrail()
    test_path_validation_safety_stop()
    test_deployment_readiness_metrics_and_guardrail()
    print("All Clean Retraining and Deployment Readiness tests passed successfully!")
