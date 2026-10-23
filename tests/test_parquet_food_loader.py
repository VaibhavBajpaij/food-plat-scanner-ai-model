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
    get_split_from_path,
    EXPECTED_43_CLASSES,
    LegacyDatasetGuardError,
    inspect_parquet_dataset,
    audit_splits_and_classes,
    audit_cross_split_leakage,
    extract_and_prepare_43class_manifest,
    ParquetFoodDataset
)
from src.classification.train_kaggle import (
    build_mobilenet_v3,
    train_indian_43class_model,
    run_actual_data_smoke_test
)

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

def test_get_split_from_path():
    # Flat filenames
    assert get_split_from_path("train_0.parquet") == "train"
    assert get_split_from_path("val_0.parquet") == "val"
    assert get_split_from_path("test_0.parquet") == "test"

    # Nested directory structures
    assert get_split_from_path("/kaggle/working/data/train/0.parquet") == "train"
    assert get_split_from_path("/kaggle/working/data/validation/0.parquet") == "val"
    assert get_split_from_path("/kaggle/working/data/test/0.parquet") == "test"
    assert get_split_from_path(r"D:\health app\datasets\raw\train\0.parquet") == "train"
    assert get_split_from_path(r"D:\health app\datasets\raw\val\0.parquet") == "val"

def test_43class_vocabulary():
    assert len(EXPECTED_43_CLASSES) == 43, f"Expected exactly 43 classes, got {len(EXPECTED_43_CLASSES)}"
    assert "masala_dosa" in EXPECTED_43_CLASSES
    assert "chole_bhature" in EXPECTED_43_CLASSES
    assert "butter_chicken" in EXPECTED_43_CLASSES
    assert "dal_makhani" in EXPECTED_43_CLASSES
    assert "biryani" in EXPECTED_43_CLASSES

def test_split_and_leakage_audit():
    records = [
        {"canonical_id": "biryani", "split": "train", "sha256": "sha_1", "perceptual_hash": "0000000000000000", "image_id": "img1"},
        {"canonical_id": "biryani", "split": "val", "sha256": "sha_2", "perceptual_hash": "ffffffffffffffff", "image_id": "img2"},
        {"canonical_id": "masala_dosa", "split": "test", "sha256": "sha_3", "perceptual_hash": "1111111111111111", "image_id": "img3"}
    ]
    split_rep = audit_splits_and_classes(records, expected_classes=["biryani", "masala_dosa"])
    assert split_rep["splits"]["train"]["classes_present_count"] == 1
    assert "masala_dosa" in split_rep["splits"]["train"]["missing_classes"]

    leakage_rep = audit_cross_split_leakage(records)
    assert leakage_rep["is_leakage_free"]
    assert leakage_rep["exact_collisions"]["train_val"] == 0

def test_legacy_dataset_guardrail_rejection():
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

def test_parquet_synthetic_roundtrip_and_smoke_test():
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
        assert audit["split_counts"]["train"] == 2
        assert "masala_dosa" in audit["classes"]

        # Extract
        out_img_dir = os.path.join(tmp_dir, "extracted")
        out_manifest = os.path.join(tmp_dir, "manifest.json")
        res = extract_and_prepare_43class_manifest(tmp_dir, out_img_dir, out_manifest)
        assert res["total_images"] == 2

        # Run smoke test on actual extracted images
        device = torch.device("cpu")
        smoke_rep = run_actual_data_smoke_test(out_manifest, batch_size=2, device=device)
        assert smoke_rep["status"] == "PASS"

def test_mobilenet_v3_43class_shape():
    device = torch.device("cpu")
    model = build_mobilenet_v3(num_classes=43, device=device)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    assert out.shape == (2, 43), f"Expected shape (2, 43), got {out.shape}"

if __name__ == "__main__":
    test_parquet_image_decoding()
    test_get_split_from_path()
    test_43class_vocabulary()
    test_split_and_leakage_audit()
    test_legacy_dataset_guardrail_rejection()
    test_parquet_synthetic_roundtrip_and_smoke_test()
    test_mobilenet_v3_43class_shape()
    print("All Parquet Food Loader, Split Preservation and Smoke Test tests passed!")
