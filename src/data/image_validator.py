"""
Image Validator and Deduplication Engine
Implements:
1. Perceptual hashing (dHash) and Hamming distance near-duplicate detection.
2. SHA256 exact checksum deduplication.
3. Content integrity validation (format, dimensions, aspect ratio, corruption).
4. Capture-session isolation validation across splits.
5. Automated diagnostic reporting.
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Tuple, Set, Optional
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_sha256(file_path: str) -> str:
    """Computes exact SHA256 checksum of an image file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_dhash(image_input, hash_size: int = 8) -> str:
    """
    Computes a 64-bit difference hash (dHash) for an image.
    Works by comparing intensity gradients horizontally across a (hash_size + 1) x hash_size grid.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input

    # Convert to grayscale and resize to (hash_size + 1, hash_size)
    img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img_gray.tobytes())

    # Compare adjacent pixels in each row
    diff = []
    width = hash_size + 1
    for row in range(hash_size):
        row_start = row * width
        for col in range(hash_size):
            p1 = pixels[row_start + col]
            p2 = pixels[row_start + col + 1]
            diff.append(p1 > p2)

    # Convert 64 bits to hex string
    decimal_val = 0
    for idx, bit in enumerate(diff):
        if bit:
            decimal_val |= 1 << idx

    return f"{decimal_val:016x}"

def hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Calculates Hamming distance between two 16-char hex perceptual hashes."""
    n1 = int(hash1_hex, 16)
    n2 = int(hash2_hex, 16)
    return bin(n1 ^ n2).count("1")

def validate_image_file(file_path: str, min_size: int = 224, max_size: int = 5000) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates image file existence, decodability, dimensions, and aspect ratio.
    Returns: (is_valid, error_reason, image_metadata)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", None

    file_size_bytes = os.path.getsize(file_path)
    if file_size_bytes < 500:  # < 500 bytes (corrupted header or truncated placeholder)
        return False, f"File size too small ({file_size_bytes} bytes), likely corrupted placeholder", None

    try:
        with Image.open(file_path) as img:
            img.verify()  # Fast structural verification

        # Re-open for geometry inspect (verify closes the file handle)
        with Image.open(file_path) as img:
            w, h = img.size
            fmt = img.format
            mode = img.mode

            if w < min_size or h < min_size:
                return False, f"Resolution {w}x{h} is below minimum {min_size}x{min_size}", None

            if w > max_size or h > max_size:
                return False, f"Resolution {w}x{h} exceeds maximum {max_size}x{max_size}", None

            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 3.2:
                return False, f"Extreme aspect ratio {aspect_ratio:.2f} (> 3.2), unsuitable for square crop", None

            meta = {
                "width": w,
                "height": h,
                "format": fmt,
                "mode": mode,
                "size_bytes": file_size_bytes,
                "aspect_ratio": round(aspect_ratio, 3)
            }
            return True, None, meta

    except Exception as e:
        return False, f"Corrupted or invalid image stream: {str(e)}", None

class DatasetIntegrityAuditor:
    """
    Audits a manifest for:
    - Missing files
    - Exact duplicate SHA256
    - Perceptual near-duplicates (dHash Hamming distance <= threshold)
    - Split and capture-group isolation
    - Class balance and license coverage
    """
    def __init__(self, manifest_path: str, near_duplicate_threshold: int = 4):
        self.manifest_path = manifest_path
        self.near_duplicate_threshold = near_duplicate_threshold

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.images = self.manifest.get("images", [])

    def run_full_audit(self) -> Dict[str, Any]:
        print("=" * 80)
        print(f"RUNNING DATASET INTEGRITY & DEDUPLICATION AUDIT: {os.path.basename(self.manifest_path)}")
        print("=" * 80)

        results = {
            "manifest_file": self.manifest_path,
            "total_images_in_manifest": len(self.images),
            "valid_images": 0,
            "corrupted_or_missing": [],
            "exact_duplicate_pairs": [],
            "near_duplicate_pairs": [],
            "split_session_violations": [],
            "per_class_counts": {},
            "split_distribution": {},
            "license_distribution": {},
            "cc0_commercial_count": 0
        }

        sha_map: Dict[str, List[Dict[str, Any]]] = {}
        hash_list: List[Tuple[str, str, str]] = []  # (image_id, file_path, dhash)
        session_to_splits: Dict[str, Set[str]] = {}

        for item in self.images:
            img_id = item.get("image_id")
            rel_path = item.get("file_path")
            full_path = os.path.join(BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
            cid = item.get("canonical_id")
            split = item.get("split", "train")
            session = item.get("capture_session_id", img_id)
            lic = item.get("license", "unknown")
            is_cc0 = item.get("is_cc0_or_public_domain", False)

            # Split tracking
            results["split_distribution"][split] = results["split_distribution"].get(split, 0) + 1
            results["per_class_counts"][cid] = results["per_class_counts"].get(cid, 0) + 1
            results["license_distribution"][lic] = results["license_distribution"].get(lic, 0) + 1
            if is_cc0:
                results["cc0_commercial_count"] += 1

            # Session leakage tracking
            if session not in session_to_splits:
                session_to_splits[session] = set()
            session_to_splits[session].add(split)

            # Integrity check
            valid, err, meta = validate_image_file(full_path)
            if not valid:
                results["corrupted_or_missing"].append({
                    "image_id": img_id,
                    "file_path": rel_path,
                    "reason": err
                })
                continue

            results["valid_images"] += 1

            # SHA256 check
            file_sha = compute_sha256(full_path)
            if file_sha not in sha_map:
                sha_map[file_sha] = []
            sha_map[file_sha].append({"image_id": img_id, "file_path": rel_path, "split": split})

            # Perceptual hash check
            dh = compute_dhash(full_path)
            hash_list.append((img_id, rel_path, dh, split))

        # Check for exact duplicate SHA256
        for sha, occurrences in sha_map.items():
            if len(occurrences) > 1:
                results["exact_duplicate_pairs"].append({
                    "sha256": sha,
                    "occurrences": occurrences
                })

        # Check for near-duplicates via dHash Hamming distance
        n = len(hash_list)
        for i in range(n):
            id1, path1, h1, s1 = hash_list[i]
            for j in range(i + 1, n):
                id2, path2, h2, s2 = hash_list[j]
                dist = hamming_distance(h1, h2)
                if dist <= self.near_duplicate_threshold:
                    results["near_duplicate_pairs"].append({
                        "image_1": id1,
                        "image_2": id2,
                        "hamming_distance": dist,
                        "splits": [s1, s2],
                        "paths": [path1, path2]
                    })

        # Check for session split leakage
        for sess, splits in session_to_splits.items():
            if len(splits) > 1:
                results["split_session_violations"].append({
                    "capture_session_id": sess,
                    "splits_contaminated": list(splits)
                })

        print(f"Results Summary:")
        print(f"  • Total Images:          {results['total_images_in_manifest']}")
        print(f"  • Valid Decodable:       {results['valid_images']}")
        print(f"  • Missing / Corrupted:   {len(results['corrupted_or_missing'])}")
        print(f"  • Exact Duplicate Pairs: {len(results['exact_duplicate_pairs'])}")
        print(f"  • Near Duplicate Pairs:  {len(results['near_duplicate_pairs'])} (Hamming Dist <= {self.near_duplicate_threshold})")
        print(f"  • Split Violations:      {len(results['split_session_violations'])}")
        print(f"  • CC0 Commercial Subset: {results['cc0_commercial_count']} images")
        print(f"  • Split Breakdown:       {results['split_distribution']}")
        print("=" * 80)

        return results

if __name__ == "__main__":
    v3_manifest = os.path.join(BASE_DIR, "data_manifests", "food_dataset_v3_manifest.json")
    if os.path.exists(v3_manifest):
        auditor = DatasetIntegrityAuditor(v3_manifest)
        res = auditor.run_full_audit()
        out_file = os.path.join(BASE_DIR, "outputs", "dataset_v3_integrity_audit.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Audit output written to: {out_file}")
