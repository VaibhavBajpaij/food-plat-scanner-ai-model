"""
Unit Tests for Parquet Food Dataset Loader & 43-Class Indian Food Pipeline
"""

import os
import sys
import io
import unittest
import tempfile
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.parquet_food_loader import (
    decode_parquet_image,
    EXPECTED_43_CLASSES,
    LegacyDatasetGuardError,
    inspect_parquet_dataset,
    extract_and_prepare_43class_manifest,
    ParquetFoodDataset
)
from src.classification.train_kaggle import build_mobilenet_v3, train_indian_43class_model

def test_parquet_image_decoding():
    # 1. Valid image in dict struct
    img = Image.new("RGB", (64, 64), color=(255, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_bytes = buf.getvalue()

    decoded = decode_parquet_image({"bytes": valid_bytes, "path": None})
    assert decoded.size == (64, 64)
    assert decoded.mode == "RGB"

    # 2. Raw bytes
    decoded_raw = decode_parquet_image(valid_bytes)
    assert decoded_raw.size == (64, 64)

    # 3. Corrupted image buffer should raise ValueError
    try:
        decode_parquet_image(b"not-a-valid-jpeg-stream")
        assert False, "Corrupted bytes should have raised ValueError!"
    except ValueError:
        pass

def test_43class_vocabulary():
    assert len(EXPECTED_43_CLASSES) == 43, f"Expected exactly 43 classes, got {len(EXPECTED_43_CLASSES)}"
    assert "masala_dosa" in EXPECTED_43_CLASSES
    assert "chole_bhature" in EXPECTED_43_CLASSES
    assert "butter_chicken" in EXPECTED_43_CLASSES
    assert "dal_makhani" in EXPECTED_43_CLASSES
    assert "biryani" in EXPECTED_43_CLASSES

def test_legacy_dataset_guardrail_rejection():
    # If a manifest has fewer than 1,000 images or fewer than 40 classes, it must fail loudly!
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json_path = f.name
        import json
        json.dump({
            "images": [
                {"canonical_id": "apple", "split": "train", "file_path": "dummy.jpg"},
                {"canonical_id": "rice", "split": "val", "file_path": "dummy2.jpg"}
            ]
        }, f)

    try:
        train_indian_43class_model(manifest_path=json_path)
        assert False, "Legacy tiny dataset must raise LegacyDatasetGuardError!"
    except LegacyDatasetGuardError:
        pass
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)

def test_parquet_synthetic_roundtrip():
    # Create synthetic parquet file with PyArrow
    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = os.path.join(tmp_dir, "train_0.parquet")
        img1 = Image.new("RGB", (64, 64), color=(100, 200, 50))
        buf1 = io.BytesIO()
        img1.save(buf1, format="JPEG")

        img2 = Image.new("RGB", (64, 64), color=(50, 100, 200))
        buf2 = io.BytesIO()
        img2.save(buf2, format="JPEG")

        images_col = [
            {"bytes": buf1.getvalue(), "path": None},
            {"bytes": buf2.getvalue(), "path": None}
        ]
        labels_col = ["masala_dosa", "butter_chicken"]

        # Struct type for images
        schema = pa.schema([
            ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("label", pa.string())
        ])

        table = pa.Table.from_arrays([
            pa.array(images_col, type=schema.field("image").type),
            pa.array(labels_col, type=pa.string())
        ], schema=schema)

        pq.write_table(table, pq_path)

        # Inspect
        audit = inspect_parquet_dataset([pq_path])
        assert audit["total_rows"] == 2
        assert "masala_dosa" in audit["classes"]
        assert "butter_chicken" in audit["classes"]

        # Extract and verify manifest
        out_img_dir = os.path.join(tmp_dir, "extracted")
        out_manifest = os.path.join(tmp_dir, "manifest.json")
        res = extract_and_prepare_43class_manifest(tmp_dir, out_img_dir, out_manifest)
        assert res["total_images"] == 2
        assert len(res["images"]) == 2

def test_mobilenet_v3_43class_shape():
    device = torch.device("cpu")
    model = build_mobilenet_v3(num_classes=43, device=device)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 43), f"Expected shape (2, 43), got {out.shape}"

if __name__ == "__main__":
    test_parquet_image_decoding()
    test_43class_vocabulary()
    test_legacy_dataset_guardrail_rejection()
    test_parquet_synthetic_roundtrip()
    test_mobilenet_v3_43class_shape()
    print("All Parquet Food Loader and 43-Class unit tests passed!")
