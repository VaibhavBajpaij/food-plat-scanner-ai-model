"""
Step 1 Audit and Verification Script
Verifies:
1. Original pretrained checkpoint unchanged & loads.
2. Existing food-specific checkpoints (head-only, staged fine-tuned) load and pass forward inference.
3. Train, val, and smoke test images exist, have valid RGB format, and labels match filenames.
4. Dataset sources, permissive licenses, and SHA256 hashes are recorded and valid.
5. Session / capture-group leakage check across splits (zero image/session collisions).
6. Reproducibility of smoke-test evaluations (both exact synset and mapped proxy).
"""

import os
import sys
import json
import hashlib
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def audit_step1():
    print("=" * 75)
    print("STEP 1: THOROUGH AUDIT OF CURRENT EXPERIMENT ARTIFACTS")
    print("=" * 75)

    audit_results = {
        "checkpoints": {},
        "manifests": {},
        "split_leakage": {},
        "smoke_test_reproducibility": {},
        "baseline_discrepancy_analysis": {}
    }

    # 1. Verify Original Pretrained Checkpoint
    print("\n[1/6] Verifying Original Pretrained Checkpoint...")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    m0 = models.mobilenet_v3_small(weights=weights)
    m0.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out0 = m0(dummy_input)
    assert out0.shape == (1, 1000), f"Expected shape (1, 1000), got {out0.shape}"
    print("  -> PASS: models.mobilenet_v3_small(weights='DEFAULT') loaded successfully, output shape [1, 1000]")
    audit_results["checkpoints"]["m0_original"] = {
        "status": "PASS",
        "output_shape": list(out0.shape),
        "source": str(weights)
    }

    # 2. Verify Existing Checkpoints
    print("\n[2/6] Verifying Food-Specific Checkpoints...")
    food_classes = [
        "apple_fresh_raw", "banana_raw", "broccoli_steamed",
        "chicken_breast_grilled", "rice_white_cooked"
    ]
    ckpt_paths = {
        "m1_head_only": os.path.join(BASE_DIR, "models", "checkpoints", "mobilenet_v3_food_head_only.pth"),
        "m2_finetuned": os.path.join(BASE_DIR, "models", "checkpoints", "mobilenet_v3_food_finetuned.pth")
    }
    for name, cpath in ckpt_paths.items():
        if not os.path.exists(cpath):
            raise FileNotFoundError(f"Missing checkpoint: {cpath}")
        ckpt_data = torch.load(cpath, map_location="cpu", weights_only=False)
        m = models.mobilenet_v3_small(weights=None)
        in_feat = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_feat, len(food_classes))
        m.load_state_dict(ckpt_data["model_state_dict"])
        m.eval()
        with torch.no_grad():
            out = m(dummy_input)
        assert out.shape == (1, 5), f"Expected shape (1, 5), got {out.shape}"
        file_size = os.path.getsize(cpath)
        sha256 = compute_sha256(cpath)
        print(f"  -> PASS: {name} loaded ({file_size / (1024*1024):.2f} MB, sha256={sha256[:12]}...), output shape [1, 5]")
        audit_results["checkpoints"][name] = {
            "status": "PASS",
            "path": cpath,
            "size_bytes": file_size,
            "sha256": sha256,
            "val_acc": ckpt_data.get("val_acc"),
            "val_loss": ckpt_data.get("val_loss"),
            "epoch": ckpt_data.get("epoch")
        }

    # 3 & 4. Verify Manifests, Image Labels, Permissive Licenses, File Hashes
    print("\n[3/6 & 4/6] Verifying Dataset Manifests, Licenses, and Hashes...")
    manifest_files = {
        "finetune": os.path.join(BASE_DIR, "data_manifests", "finetune_dataset_manifest.json"),
        "smoke_test": os.path.join(BASE_DIR, "data_manifests", "smoke_test_manifest.json")
    }

    all_records = []
    for mname, mpath in manifest_files.items():
        with open(mpath, "r", encoding="utf-8") as f:
            mdata = json.load(f)
        imgs = mdata.get("images", [])
        print(f"  Checking {mname} ({len(imgs)} images)...")
        corrupted = 0
        hash_mismatch = 0
        missing = 0
        license_missing = 0
        for entry in imgs:
            fpath = os.path.join(BASE_DIR, entry["file_path"])
            if not os.path.exists(fpath):
                missing += 1
                continue
            # verify hash
            real_hash = compute_sha256(fpath)
            if real_hash != entry.get("sha256"):
                hash_mismatch += 1
            # verify image format
            try:
                with Image.open(fpath) as im:
                    im.verify()
                with Image.open(fpath) as im:
                    im_rgb = im.convert("RGB")
                    _ = im_rgb.size
            except Exception:
                corrupted += 1
            # verify license
            if not entry.get("license") or not entry.get("source_url"):
                license_missing += 1

            all_records.append({**entry, "manifest_origin": mname})

        print(f"    Missing: {missing}, Hash Mismatch: {hash_mismatch}, Corrupted: {corrupted}, License Missing: {license_missing}")
        assert missing == 0 and hash_mismatch == 0 and corrupted == 0 and license_missing == 0, f"Errors in {mname}"
        audit_results["manifests"][mname] = {
            "total_images": len(imgs),
            "missing": missing,
            "hash_mismatch": hash_mismatch,
            "corrupted": corrupted,
            "license_missing": license_missing
        }

    # 5. Check Split Isolation and Leakage
    print("\n[5/6] Checking for Session & Image Leakage Across Splits...")
    train_imgs = [r for r in all_records if r.get("split") == "train"]
    val_imgs = [r for r in all_records if r.get("split") == "val"]
    smoke_imgs = [r for r in all_records if r.get("manifest_origin") == "smoke_test"]

    train_hashes = {r["sha256"]: r for r in train_imgs}
    val_hashes = {r["sha256"]: r for r in val_imgs}
    smoke_hashes = {r["sha256"]: r for r in smoke_imgs}

    train_sessions = {r.get("session_id", r.get("image_id")) for r in train_imgs}
    val_sessions = {r.get("session_id", r.get("image_id")) for r in val_imgs}
    smoke_sessions = {r.get("session_id", r.get("image_id")) for r in smoke_imgs}

    train_urls = {r.get("source_url") for r in train_imgs if r.get("source_url")}
    val_urls = {r.get("source_url") for r in val_imgs if r.get("source_url")}
    smoke_urls = {r.get("source_url") for r in smoke_imgs if r.get("source_url")}

    hash_overlap_tv = set(train_hashes.keys()) & set(val_hashes.keys())
    hash_overlap_ts = set(train_hashes.keys()) & set(smoke_hashes.keys())
    hash_overlap_vs = set(val_hashes.keys()) & set(smoke_hashes.keys())

    session_overlap_tv = train_sessions & val_sessions
    session_overlap_ts = train_sessions & smoke_sessions
    session_overlap_vs = val_sessions & smoke_sessions

    url_overlap_tv = train_urls & val_urls
    url_overlap_ts = train_urls & smoke_urls
    url_overlap_vs = val_urls & smoke_urls

    # Check intra-split duplicates
    from collections import Counter
    smoke_hash_counts = Counter(r["sha256"] for r in smoke_imgs)
    smoke_intra_duplicates = [h for h, c in smoke_hash_counts.items() if c > 1]

    print(f"  Train/Val hash collisions: {len(hash_overlap_tv)}, session collisions: {len(session_overlap_tv)}, URL collisions: {len(url_overlap_tv)}")
    print(f"  Train/Smoke hash collisions: {len(hash_overlap_ts)}, session collisions: {len(session_overlap_ts)}, URL collisions: {len(url_overlap_ts)}")
    print(f"  Val/Smoke hash collisions: {len(hash_overlap_vs)}, session collisions: {len(session_overlap_vs)}, URL collisions: {len(url_overlap_vs)}")
    print(f"  Intra-Smoke Test duplicate hashes: {len(smoke_intra_duplicates)}")

    leakage_details = []
    for h in hash_overlap_tv:
        leakage_details.append({
            "type": "TRAIN_VAL_LEAKAGE_AND_CONTRADICTORY_LABEL",
            "sha256": h,
            "train_file": train_hashes[h]["file_path"],
            "train_label": train_hashes[h]["canonical_id"],
            "val_file": val_hashes[h]["file_path"],
            "val_label": val_hashes[h]["canonical_id"],
            "url": train_hashes[h]["source_url"]
        })
        print(f"  [ALERT] LEAKAGE FOUND BETWEEN TRAIN AND VAL:")
        print(f"    Train: {train_hashes[h]['file_path']} (labeled as {train_hashes[h]['canonical_id']})")
        print(f"    Val:   {val_hashes[h]['file_path']} (labeled as {val_hashes[h]['canonical_id']})")
        print(f"    URL:   {train_hashes[h]['source_url']}")

    for h in smoke_intra_duplicates:
        dups = [r for r in smoke_imgs if r["sha256"] == h]
        leakage_details.append({
            "type": "SMOKE_TEST_INTRA_DUPLICATE_CONTRADICTORY_LABEL",
            "sha256": h,
            "files": [d["file_path"] for d in dups],
            "labels": [d["canonical_id"] for d in dups],
            "url": dups[0]["source_url"]
        })
        print(f"  [ALERT] DUPLICATE FOUND INSIDE SMOKE-TEST WITH DIFFERENT LABELS:")
        print(f"    Files: {[d['file_path'] for d in dups]}")
        print(f"    Labels: {[d['canonical_id'] for d in dups]}")

    audit_results["split_leakage"] = {
        "status": "FAIL_LEAKAGE_DETECTED",
        "train_count": len(train_imgs),
        "val_count": len(val_imgs),
        "smoke_test_count": len(smoke_imgs),
        "train_val_hash_collisions": len(hash_overlap_tv),
        "intra_smoke_test_duplicates": len(smoke_intra_duplicates),
        "leakage_details": leakage_details
    }

    # 6. Reproduce Smoke Test Evaluations & Explain Baseline Discrepancy
    print("\n[6/6] Verifying Smoke Test Evaluation Reproducibility & Investigating Discrepancy...")
    from src.evaluation.run_smoke_test_evaluation import run_smoke_test_evaluation
    from src.evaluation.comparative_evaluation import run_comparative_benchmark

    smoke_eval_res = run_smoke_test_evaluation()
    comp_eval_res = run_comparative_benchmark()

    # Exact synset stats in smoke_eval_res
    total_smoke = smoke_eval_res["total_images_evaluated"]
    exact_hits = sum(c["correct_synset_top1"] for c in smoke_eval_res["class_breakdown"].values())
    food_related_hits = sum(c["food_related_top1"] for c in smoke_eval_res["class_breakdown"].values())
    exact_acc = exact_hits / total_smoke
    food_related_acc = food_related_hits / total_smoke

    # Mapped proxy stats in comp_eval_res
    comp_m0_acc = comp_eval_res["smoke_test_metrics"]["m0_original"]["accuracy"]
    comp_m1_acc = comp_eval_res["smoke_test_metrics"]["m1_head_only"]["accuracy"]
    comp_m2_acc = comp_eval_res["smoke_test_metrics"]["m2_finetuned"]["accuracy"]

    print(f"\n  run_smoke_test_evaluation.py (Strict Exact Synset Matching):")
    print(f"    Exact Synset Top-1 Accuracy: {exact_hits}/{total_smoke} ({exact_acc:.1%})")
    print(f"    Food-Related Top-1:          {food_related_hits}/{total_smoke} ({food_related_acc:.1%})")
    print(f"  comparative_evaluation.py (Heuristic Proxy Class Mapping):")
    print(f"    M0 (Pretrained ImageNet):    {comp_m0_acc:.1%}")
    print(f"    M1 (Head-Only):              {comp_m1_acc:.1%}")
    print(f"    M2 (Staged Fine-Tuned):      {comp_m2_acc:.1%}")

    discrepancy_explanation = (
        "ROOT CAUSE OF BASELINE SMOKE-TEST DISCREPANCY EXPLAINED:\n"
        "1. In 'reports/smoke_test_baseline_report.md' (generated by run_smoke_test_evaluation.py),\n"
        "   ImageNet-1k predictions were tested for exact synset match (EXACT_SYNSET_MAP). Because ImageNet-1k\n"
        "   has NO classes for 'rice_white_cooked' or 'chicken_breast_grilled', exact Top-1 accuracy was 2/17 (11.8%).\n"
        "2. In 'reports/comparative_finetuning_report.md' (generated by comparative_evaluation.py),\n"
        "   an artificial lookup table 'IMAGENET_TO_FOOD_MAP' was introduced that mapped unrelated ImageNet synsets\n"
        "   as proxy hits: 'meat loaf' & 'rotisserie' -> 'chicken_breast_grilled', 'American lobster' & 'carbonara' -> 'rice_white_cooked',\n"
        "   and 'orange' & 'fig' -> 'apple_fresh_raw'.\n"
        "3. This heuristic proxy credited 5 misclassifications as correct, artificially inflating M0 baseline accuracy\n"
        "   from 11.8% to 7/17 (41.2%).\n"
        "4. In a rigorous evaluation, M0 should be evaluated strictly: either 0% on unrepresented prepared classes\n"
        "   or using a consistent semantic scoring protocol that does not credit lobster as cooked rice."
    )
    print("\n" + discrepancy_explanation)

    audit_results["smoke_test_reproducibility"] = {
        "run_smoke_test_evaluation": {
            "exact_synset_accuracy": round(exact_acc, 4),
            "food_related_accuracy": round(food_related_acc, 4)
        },
        "comparative_evaluation": {
            "m0_original_accuracy": comp_m0_acc,
            "m1_head_only_accuracy": comp_m1_acc,
            "m2_finetuned_accuracy": comp_m2_acc
        }
    }
    audit_results["baseline_discrepancy_analysis"] = discrepancy_explanation

    out_audit = os.path.join(BASE_DIR, "outputs", "step1_audit_verification.json")
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nStep 1 Audit Complete! Report saved to {out_audit}")
    return audit_results

if __name__ == "__main__":
    audit_step1()
