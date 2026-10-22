"""
Curate and Download the 50-Image Controlled Fine-Tuning Dataset
Classes: rice_white_cooked, chicken_breast_grilled, broccoli_steamed, apple_fresh_raw, banana_raw.
Strictly excludes all 17 smoke-test images and groups images by session to prevent leakage.
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.parse
from PIL import Image

def get_smoke_test_exclusions(base_dir: str):
    manifest_path = os.path.join(base_dir, "data_manifests", "smoke_test_manifest.json")
    excluded_hashes = set()
    excluded_urls = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("images", []):
                excluded_hashes.add(item.get("sha256"))
                excluded_urls.add(item.get("source_url"))
    return excluded_hashes, excluded_urls

def search_wikimedia(query: str, limit: int = 25):
    api_url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit={limit}"
        "&prop=imageinfo&iiprop=url|size|mime|extmetadata&format=json"
    )
    headers = {
        "User-Agent": "FoodScannerResearchPipeline/1.0 (academic research benchmark; contact: bajpaivaibhav2005@gmail.com)"
    }
    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        results = []
        for pid, p in pages.items():
            ii_list = p.get("imageinfo", [])
            if not ii_list:
                continue
            ii = ii_list[0]
            if ii.get("mime") not in ["image/jpeg", "image/png"]:
                continue
            meta = ii.get("extmetadata", {})
            results.append({
                "title": p.get("title", ""),
                "url": ii.get("url", ""),
                "license": meta.get("LicenseShortName", {}).get("value", "Public Domain / CC"),
                "author": meta.get("Artist", {}).get("value", "Unknown"),
            })
        return results
    except Exception as e:
        print(f"Error querying '{query}': {e}", file=sys.stderr)
        return []

def download_and_process(url: str, dest_path: str) -> bool:
    headers = {
        "User-Agent": "FoodScannerResearchPipeline/1.0 (academic research benchmark; contact: bajpaivaibhav2005@gmail.com)"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        with Image.open(dest_path) as img:
            img.verify()
        with Image.open(dest_path) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
            rgb.save(dest_path, "JPEG", quality=90)
        return True
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def build_finetune_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    target_dir = os.path.join(base_dir, "datasets", "finetune")
    manifest_path = os.path.join(base_dir, "data_manifests", "finetune_dataset_manifest.json")
    os.makedirs(os.path.join(target_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "val"), exist_ok=True)

    excluded_hashes, excluded_urls = get_smoke_test_exclusions(base_dir)

    CLASSES = [
        {"cid": "rice_white_cooked", "query": "cooked rice plate bowl", "train": 8, "val": 2},
        {"cid": "chicken_breast_grilled", "query": "grilled chicken meat food plate", "train": 8, "val": 2},
        {"cid": "broccoli_steamed", "query": "cooked broccoli dish food", "train": 8, "val": 2},
        {"cid": "apple_fresh_raw", "query": "apple fruit table plate", "train": 8, "val": 2},
        {"cid": "banana_raw", "query": "banana fruit fresh ripe", "train": 8, "val": 2},
    ]

    manifest_records = []
    total_downloaded = 0

    print("=" * 70)
    print("CURATING CONTROLLED 50-IMAGE FINE-TUNING DATASET")
    print(f"Excluding {len(excluded_hashes)} smoke test hashes to prevent contamination.")
    print("=" * 70)

    for item in CLASSES:
        cid = item["cid"]
        q = item["query"]
        total_needed = item["train"] + item["val"]
        print(f"\nSearching for class '{cid}' (target: {total_needed} images)...")
        candidates = search_wikimedia(q, limit=30)

        acquired = 0
        for cand in candidates:
            if acquired >= total_needed:
                break
            if cand["url"] in excluded_urls:
                continue

            split = "train" if acquired < item["train"] else "val"
            idx = (acquired + 1) if split == "train" else (acquired - item["train"] + 1)
            filename = f"{cid}_{split}_{idx:02d}.jpg"
            dest_file = os.path.join(target_dir, split, filename)
            rel_path = f"datasets/finetune/{split}/{filename}"

            success = download_and_process(cand["url"], dest_file)
            if not success:
                continue

            with open(dest_file, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()

            if sha in excluded_hashes:
                os.remove(dest_file)
                continue

            record = {
                "image_id": f"ft_{cid}_{split}_{idx:02d}",
                "file_path": rel_path,
                "canonical_id": cid,
                "split": split,
                "source_url": cand["url"],
                "wikimedia_title": cand["title"],
                "license": cand["license"],
                "author": cand["author"][:60] if cand["author"] else "Unknown",
                "session_id": f"session_{cid}_{split}_{idx:02d}",
                "sha256": sha
            }
            manifest_records.append(record)
            acquired += 1
            total_downloaded += 1
            print(f"  [{acquired}/{total_needed}] [{split.upper()}] {filename} ({cand['license']})")

    manifest = {
        "dataset_name": "Controlled Food Fine-Tuning Dataset (5 Classes)",
        "total_images": total_downloaded,
        "classes": [c["cid"] for c in CLASSES],
        "split_summary": {
            "train": sum(1 for r in manifest_records if r["split"] == "train"),
            "val": sum(1 for r in manifest_records if r["split"] == "val"),
            "smoke_test_independent": 17
        },
        "per_class_summary": {c["cid"]: sum(1 for r in manifest_records if r["canonical_id"] == c["cid"]) for c in CLASSES},
        "license_breakdown": {
            "CC0 / Public Domain / CC-BY / CC-BY-SA": total_downloaded
        },
        "images": manifest_records
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Successfully acquired {total_downloaded} images.")
    print(f"Train: {manifest['split_summary']['train']} | Val: {manifest['split_summary']['val']}")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 70)
    return manifest

if __name__ == "__main__":
    build_finetune_dataset()
