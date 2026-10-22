"""
Balance validation splits so each class has exactly 2 validation images.
Total: 22 train, 10 val, and 17 external smoke test images.
"""

import os
import json
import shutil

def balance_dataset_splits():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    manifest_p = os.path.join(base_dir, "data_manifests", "finetune_dataset_manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Group images by class
    by_class = {}
    for img in manifest["images"]:
        cid = img["canonical_id"]
        by_class.setdefault(cid, []).append(img)

    updated_images = []
    train_dir = os.path.join(base_dir, "datasets", "finetune", "train")
    val_dir = os.path.join(base_dir, "datasets", "finetune", "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for cid, items in by_class.items():
        # Last 2 items become val
        val_items = items[-2:]
        train_items = items[:-2]

        for idx, item in enumerate(train_items):
            old_path = os.path.join(base_dir, item["file_path"])
            new_rel = f"datasets/finetune/train/{cid}_train_{idx+1:02d}.jpg"
            new_path = os.path.join(base_dir, new_rel)
            if old_path != new_path and os.path.exists(old_path):
                shutil.move(old_path, new_path)
            item["split"] = "train"
            item["file_path"] = new_rel
            item["image_id"] = f"ft_{cid}_train_{idx+1:02d}"
            updated_images.append(item)

        for idx, item in enumerate(val_items):
            old_path = os.path.join(base_dir, item["file_path"])
            new_rel = f"datasets/finetune/val/{cid}_val_{idx+1:02d}.jpg"
            new_path = os.path.join(base_dir, new_rel)
            if old_path != new_path and os.path.exists(old_path):
                shutil.move(old_path, new_path)
            item["split"] = "val"
            item["file_path"] = new_rel
            item["image_id"] = f"ft_{cid}_val_{idx+1:02d}"
            updated_images.append(item)

    manifest["images"] = updated_images
    manifest["total_images"] = len(updated_images)
    manifest["split_summary"] = {
        "train": sum(1 for r in updated_images if r["split"] == "train"),
        "val": sum(1 for r in updated_images if r["split"] == "val"),
        "smoke_test_independent": 17
    }
    manifest["per_class_summary"] = {
        cid: {
            "train": sum(1 for r in updated_images if r["canonical_id"] == cid and r["split"] == "train"),
            "val": sum(1 for r in updated_images if r["canonical_id"] == cid and r["split"] == "val")
        } for cid in by_class
    }

    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print("BALANCED DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total Images: {manifest['total_images']}")
    print(f"Train: {manifest['split_summary']['train']} | Val: {manifest['split_summary']['val']}")
    for cid, cnts in manifest["per_class_summary"].items():
        print(f"  {cid:24} Train: {cnts['train']} | Val: {cnts['val']}")
    print("=" * 60)

if __name__ == "__main__":
    balance_dataset_splits()
