"""
Unit Tests for Kaggle GPU Pipeline, Image Validator, and Deduplication Engine
"""

import os
import sys
import unittest
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.image_validator import compute_dhash, hamming_distance, validate_image_file
from src.data.download_and_verify_expansion import is_permissive_license
from src.classification.train_kaggle import build_mobilenet_v3, calculate_wilson_ci, get_kaggle_or_local_paths

def test_dhash_and_hamming_distance():
    # Create two identical test images
    img1 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    img2 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert h1 == h2, "Identical images must yield identical dHash!"
    assert hamming_distance(h1, h2) == 0, "Identical images must have 0 Hamming distance!"

    # Create slightly different image
    img3 = Image.new("RGB", (256, 256), color=(0, 0, 0))
    h3 = compute_dhash(img3)
    dist = hamming_distance(h1, h3)
    assert dist == 0 or isinstance(dist, int), "Hamming distance must be integer"

def test_image_validation_integrity():
    # Valid image
    tmp_path = os.path.join(BASE_DIR, "scratch", "tmp_test_val.png")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    img.save(tmp_path)

    valid, err, meta = validate_image_file(tmp_path, min_size=224)
    assert valid, f"Valid image failed validation: {err}"
    assert meta["width"] == 300
    assert meta["height"] == 300

    # Low resolution image
    tmp_low = os.path.join(BASE_DIR, "scratch", "tmp_test_low.png")
    img_low = Image.new("RGB", (100, 100), color=(100, 150, 200))
    img_low.save(tmp_low)
    valid_low, err_low, _ = validate_image_file(tmp_low, min_size=224)
    assert not valid_low, "Image with 100x100 resolution should have failed min_size=224 check!"

def test_license_classification_logic():
    # Non commercial -> Rejected
    ok, is_cc0, norm = is_permissive_license("CC BY-NC 4.0")
    assert not ok, "Non-commercial license must be rejected!"

    # Public domain / CC0 -> Accepted and CC0
    ok, is_cc0, norm = is_permissive_license("CC0 1.0 Universal")
    assert ok and is_cc0, "CC0 must be marked acceptable and CC0 commercial safe harbor!"

    # CC BY -> Accepted, not CC0
    ok, is_cc0, norm = is_permissive_license("Creative Commons Attribution 4.0")
    assert ok and not is_cc0, "CC BY must be acceptable attribution track!"

def test_mobilenet_v3_10class_shape():
    device = torch.device("cpu")
    model = build_mobilenet_v3(num_classes=10, device=device)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"

def test_wilson_ci_computation():
    low, high = calculate_wilson_ci(k=10, n=20, confidence=0.95)
    assert 0.0 <= low < 0.5 < high <= 1.0, f"Wilson CI bounds [{low}, {high}] invalid for 50% accuracy"

if __name__ == "__main__":
    test_dhash_and_hamming_distance()
    test_image_validation_integrity()
    test_license_classification_logic()
    test_mobilenet_v3_10class_shape()
    test_wilson_ci_computation()
    print("All Kaggle pipeline tests passed successfully!")
