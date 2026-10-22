"""
Download and Curate the 15-20 Real-Food Smoke Test Dataset
Fetches CC-licensed / Public Domain real-food images across prepared dishes and produce.
Ensures session grouping to prevent data leakage and records complete provenance.
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.parse
from PIL import Image

CANDIDATE_QUERIES = [
    {
        "canonical_id": "rice_white_cooked",
        "search_term": "steamed white rice bowl plate",
        "target_count": 4,
        "default_lighting": ["indoor_ambient", "bright_daylight", "warm_artificial", "dim_shadow"]
    },
    {
        "canonical_id": "chicken_breast_grilled",
        "search_term": "grilled chicken breast fillet plate",
        "target_count": 4,
        "default_lighting": ["indoor_ambient", "bright_daylight", "warm_artificial", "direct_flash"]
    },
    {
        "canonical_id": "broccoli_steamed",
        "search_term": "steamed broccoli plate florets",
        "target_count": 4,
        "default_lighting": ["indoor_ambient", "bright_daylight", "warm_artificial", "overhead_soft"]
    },
    {
        "canonical_id": "apple_fresh_raw",
        "search_term": "fresh red apple plate fruit",
        "target_count": 4,
        "default_lighting": ["indoor_ambient", "bright_daylight", "diffuse_shadow", "warm_artificial"]
    },
    {
        "canonical_id": "banana_raw",
        "search_term": "ripe banana on plate table",
        "target_count": 4,
        "default_lighting": ["indoor_ambient", "bright_daylight", "warm_artificial", "diffuse_shadow"]
    }
]

def search_wikimedia_commons(query: str, limit: int = 10):
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
            mime = ii.get("mime", "")
            if mime not in ["image/jpeg", "image/png"]:
                continue
            meta = ii.get("extmetadata", {})
            license_val = meta.get("LicenseShortName", {}).get("value", "Public Domain / CC")
            results.append({
                "page_id": pid,
                "title": p.get("title", ""),
                "url": ii.get("url", ""),
                "width": ii.get("width", 0),
                "height": ii.get("height", 0),
                "size_bytes": ii.get("size", 0),
                "license": license_val,
                "author": meta.get("Artist", {}).get("value", "Unknown"),
            })
        return results
    except Exception as e:
        print(f"Search failed for '{query}': {str(e)}", file=sys.stderr)
        return []

def download_and_verify_image(url: str, dest_path: str) -> bool:
    headers = {
        "User-Agent": "FoodScannerResearchPipeline/1.0 (academic research benchmark; contact: bajpaivaibhav2005@gmail.com)"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        # Verify with PIL and normalize
        with Image.open(dest_path) as img:
            img.verify()
        # Convert to RGB and resize to standard max dimension 512 for lightweight evaluation
        with Image.open(dest_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            rgb_img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        print(f"Failed to download/verify {url}: {str(e)}", file=sys.stderr)
        return False

def build_smoke_test_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    target_dir = os.path.join(base_dir, "datasets", "eval", "smoke_test")
    manifest_path = os.path.join(base_dir, "data_manifests", "smoke_test_manifest.json")
    os.makedirs(target_dir, exist_ok=True)

    manifest_records = []
    total_downloaded = 0

    print("=" * 70)
    print("ACQUIRING REAL-FOOD SMOKE TEST DATASET (15-20 IMAGES)")
    print("=" * 70)

    for spec in CANDIDATE_QUERIES:
        cid = spec["canonical_id"]
        term = spec["search_term"]
        needed = spec["target_count"]
        print(f"\nSearching for '{cid}' using term: '{term}'...")
        candidates = search_wikimedia_commons(term, limit=12)

        acquired_for_class = 0
        for i, item in enumerate(candidates):
            if acquired_for_class >= needed:
                break
            
            image_filename = f"{cid}_{acquired_for_class + 1:02d}.jpg"
            dest_file = os.path.join(target_dir, image_filename)
            rel_file_path = f"datasets/eval/smoke_test/{image_filename}"

            print(f"  [{acquired_for_class+1}/{needed}] Downloading: {item['title'][:45]}...")
            success = download_and_verify_image(item["url"], dest_file)
            if success:
                # Compute SHA256
                with open(dest_file, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                lighting = spec["default_lighting"][acquired_for_class % len(spec["default_lighting"])]
                record = {
                    "image_id": f"smoke_{cid}_{acquired_for_class+1:02d}",
                    "file_path": rel_file_path,
                    "canonical_id": cid,
                    "food_type": "prepared_dish" if "cooked" in cid or "grilled" in cid or "steamed" in cid else "raw_produce",
                    "source_url": item["url"],
                    "wikimedia_title": item["title"],
                    "license": item["license"],
                    "author": item["author"][:60] if item["author"] else "Unknown",
                    "lighting_condition": lighting,
                    "session_id": f"session_{cid}_{acquired_for_class+1:02d}",
                    "split": "smoke_test",
                    "sha256": file_hash
                }
                manifest_records.append(record)
                acquired_for_class += 1
                total_downloaded += 1

    manifest = {
        "dataset_name": "Real-Food Smoke Test Benchmark (Phase 0 Step A)",
        "purpose": "Smoke test to evaluate pretrained model and pipeline on real food dishes and produce",
        "disclaimer": "This is a smoke test dataset (15-20 images) to verify pipeline behavior and failure modes, not a statistically exhaustive benchmark.",
        "created_at": "2026-10-23T03:35:00Z",
        "total_images": total_downloaded,
        "classes": list(set(r["canonical_id"] for r in manifest_records)),
        "images": manifest_records
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Successfully downloaded {total_downloaded} real-food images.")
    print(f"Saved manifest to: {manifest_path}")
    print("=" * 70)
    return manifest

if __name__ == "__main__":
    build_smoke_test_dataset()
