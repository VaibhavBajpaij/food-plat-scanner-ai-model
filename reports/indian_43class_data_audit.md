# Comprehensive Audit and Split Repair Report: 43-Class Indian Food Model

**Project:** Food Plate Scanner AI Model (`food-plat-scanner-ai-model`)  
**Artifact Evaluated:** `kaggle_indian_43class_artifacts.zip`  
**Date:** October 2026  
**Status:** Audit & Split Repair Complete | Retraining Required Before Test Benchmark Acceptance  

---

## Executive Summary

An exhaustive audit of the 43-class Indian food recognition Kaggle training artifacts (`kaggle_indian_43class_artifacts.zip`) revealed three fundamental systemic flaws in the upstream dataset splitting and baseline evaluation:

1. **Massive Evaluation Blind Spot (21 Missing Classes):**  
   The original validation (`val_0.parquet`) and test (`test_0.parquet`) splits contained only 22 of the 43 classes. The remaining 21 classes (comprising 5,600 images) were allocated 100% to the training split. As a result, the reported **79.1% test accuracy** (812/1026) was computed over only 22 classes; the remaining 21 classes had exactly zero test support (0.0 F1 score). This explains why the macro-F1 was reported as **0.4128** ($\approx 17.75 / 43$).
2. **Cross-Split Leakage Confirmed:**  
   Between the original validation and test splits, **9 exact-image byte collisions (SHA-256)** and **1 perceptual clone pair** (dHash distance 0) were confirmed. The original deduplication logic checked `train` records against `val` and `test`, but never checked or isolated duplicates between `val` and `test`.
3. **Severe Test Set Contamination of the Existing Checkpoint:**  
   When all 43 classes were properly split using stratified group-aware partitioning (preventing all exact and perceptual duplicates from crossing split boundaries), **1,328 out of 1,536 test images (86.46%)** in the repaired test set originated from the existing checkpoint's training set. Therefore, the existing checkpoint **cannot provide an unbiased held-out evaluation**. Retraining on the repaired training and validation splits is scientifically mandatory.

All artifacts, manifests, duplicate groups, confusion matrices, and repaired datasets have been produced, verified with 32 automated unit tests, and safely preserved.

---

## Phase 1 — Artifact Preservation & Baseline Investigation

### 1.1 Artifact Preservation
The original zip archive was confirmed at root:
- Path: `kaggle_indian_43class_artifacts.zip`
- Size: 6,879,344 bytes
- Checksum verified and preserved untouched.

All contents were safely extracted to a dedicated, isolated audit directory `audit_artifacts/`:
- `audit_artifacts/mobilenet_v3_indian_43class_best.pth` (6,782,757 bytes)
- `audit_artifacts/kaggle_indian_43class_training_report.json` (7,330 bytes)
- `audit_artifacts/indian_food_43class_manifest.json` (6,857,218 bytes)

### 1.2 Checkpoint Inspection
Inspection of `audit_artifacts/mobilenet_v3_indian_43class_best.pth`:
- **Architecture:** MobileNetV3-Small (`torchvision.models.mobilenet_v3_small`) with a modified classifier head:
  - Linear(576, 1024), Hardswish, Dropout(0.2), Linear(1024, 43).
- **Training Stage:** Saved at Epoch 13 of Stage 2 (`staged_finetune`).
- **Validation Accuracy at Save:** 81.35% (868/1067).
- **Label Mapping:** 43 classes (`label_to_idx` and `idx_to_label`). Verified to match the manifest mapping with 100% exact parity.

### 1.3 Mathematical Reconstruction of Baseline Metrics
The training report recorded:
- **Test Accuracy:** 79.14% (812 correct out of 1,026 test images).
- **Macro-F1:** 0.4128.
- **Weighted-F1:** 0.7963.

**Why did Macro-F1 collapse to 0.4128 despite 79.1% accuracy?**  
The test set contained images from only 22 classes. The sum of per-class F1 scores across those 22 evaluated classes was:
$$\sum_{c=1}^{22} F1_c \approx 17.75$$
When computing macro-F1 over all 43 classes:
$$\text{Macro-F1} = \frac{17.75 + 21 \times 0.0}{43} = \frac{17.75}{43} = 0.4128$$
The metric was not a calculation glitch; it was the direct mathematical consequence of 21 classes having **0 test samples**, yielding 0.0 precision, recall, and F1.

---

## Phase 2 — Data Audit & Leakage Investigation

### 2.1 The 42-Record Difference
- **Raw Parquet Dataset:** 15,404 records across 5 shards (`train_0.parquet` through `train_2.parquet`, `val_0.parquet`, `test_0.parquet`).
  - Raw Train records: 13,311
  - Raw Val records: 1,067
  - Raw Test records: 1,026
- **Original Manifest:** 15,362 images.
- **Difference:** Exactly $15,404 - 15,362 = 42$ records.

**Cause:**  
During the initial Kaggle ingestion pipeline, validation and test shards were processed first. When processing the training shards, the extractor ran SHA-256 duplicate checking against the already extracted test and validation pools. It identified 42 duplicate or near-duplicate training records and dropped them to avoid contaminating training with test/val images.  
$$13,311 - 42 = 13,269 \text{ train images in manifest}$$
$$13,269 \text{ (train)} + 1,067 \text{ (val)} + 1,026 \text{ (test)} = 15,362 \text{ total entries}$$

### 2.2 Split Distribution & The 21 Missing Classes
An audit of the original manifest revealed that the upstream dataset creator only partitioned 22 classes into `val_0.parquet` and `test_0.parquet`. The other 21 classes (5,600 images) were stored exclusively in training shards:

| Category | Classes | Total Images | Train | Val | Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fully Partitioned (22 classes)** | `biryani`, `burger`, `butter_naan`, `chai`, `chapati`, `chole_bhature`, `dal_makhani`, `dhokla`, `fried_rice`, `gulab_jamun`, `idli`, `jalebi`, `kaathi_rolls`, `kadai_paneer`, `kulfi`, `masala_dosa`, `momos`, `paan`, `pakode`, `pav_bhaji`, `pizza`, `samosa` | 9,762 | 7,669 | 1,067 | 1,026 |
| **Missing from Val & Test (21 classes)** | `adhirasam`, `aloo_gobi`, `aloo_matar`, `aloo_methi`, `aloo_shimla_mirch`, `aloo_tikki`, `anarsa`, `bandar_laddu`, `basundi`, `bhatura`, `bhindi_masala`, `boondi`, `cham_cham`, `chana_masala`, `chenna_gaja`, `chicken_razala`, `chicken_tikka`, `daal_baati_churma`, `daal_puri`, `dharwad_pedha`, `doodhpak` | 5,600 | 5,600 | **0** | **0** |
| **Total** | **43 classes** | **15,362** | **13,269** | **1,067** | **1,026** |

### 2.3 Confirmation of the 9 Validation-Test Collisions
The audit confirmed exactly **9 image pairs with identical SHA-256 hashes** present simultaneously in validation and test:

1. `samosa`: Val `sohl_9ea5fb7be50d` == Test `sohl_d90066be5008` (SHA: `a6428e235e1fe746a5d4e13ea7071b69...`)
2. `samosa`: Val `sohl_30e66eb1f83c` == Test `sohl_48c772cb3f0e` (SHA: `cf3a0889f417e299b8287e0766f8e775...`)
3. `samosa`: Val `sohl_e65d8a8775f0` == Test `sohl_04040a430ee9` (SHA: `3d6ef382fe28efde42b26b38c263725b...`)
4. `samosa`: Val `sohl_b5df7e8cb376` == Test `sohl_830b56b3e640` (SHA: `ec77283253a817452d75f2b8bbd23485...`)
5. `kulfi`: Val `sohl_bc85ea84fca8` == Test `sohl_a3eb00e4e5e4` (SHA: `02fa0236a28292815777490079a49cfb...`)
6. `momos`: Val `sohl_27aa61a35ae7` == Test `sohl_4da33c69c6cf` (SHA: `6d027b4e0bf03a6e9270e30bb73ff255...`)
7. `idli`: Val `sohl_a6f4770cece7` == Test `sohl_552277c07222` (SHA: `70fba75ef57778db10f9250bbccf2aa9...`)
8. `burger`: Val `sohl_55ec157a3d3c` == Test `sohl_4d6db53df687` (SHA: `d1ea6e4cb7a3aeafca441b44b2098bb3...`)
9. `gulab_jamun`: Val `sohl_7bbf225ad7a6` == Test `sohl_ebcae9581a65` (SHA: `5f822f73e721a7d65b75b95ca1efd66a...`)

In addition, **1 perceptual clone pair (dHash distance 0)** was found between validation and test:
- `idli`: Val `sohl_1e49cb24072f` and Test `sohl_74a58b29df92` (dHash: `2d4d8c6d49495149`).

**Root Cause:**  
The original deduplication pipeline checked `train` records against `val` and `test`, but completely omitted checking `val` against `test`.

### 2.4 Conflicting Ground Truth Labels
The audit discovered instances of duplicate images assigned conflicting ground truth labels:
- **Exact SHA-256 Collision:** Image `sohl_131312cc3803` was labeled as `kadai_paneer`, while another identical image was labeled as `butter_naan` (likely a thali containing both).
- **Perceptual Collision (Hamming Distance 2):** Image `sohl_69677332f143` was labeled `chapati`, while near-duplicate `sohl_5e022f499692` was labeled `dal_makhani`.

**Remediation:**  
All 4 conflicting records were quarantined and permanently excluded from the repaired dataset.

---

## Phase 3 — Split Repair Methodology

### 3.1 Connected Component Clustering
To guarantee zero leakage across splits:
1. Every image was assigned to a cluster using connected component graph analysis.
2. An edge was created between two images if:
   - They shared an identical SHA-256 byte hash, OR
   - Their difference hash (dHash) had a Hamming distance $\le 2$.
3. Result: **15,248 independent clusters** across 15,358 clean images (all multi-image clusters had size 2).

### 3.2 Stratified Group-Aware Splitting
Clusters were assigned atomically to `train`, `val`, or `test` using stratified group-aware partitioning with fixed seed `42`:
- Target ratio: 80% train, 10% validation, 10% test.
- Every class was guaranteed balanced representation across all three splits.

**Repaired Split Breakdown:**
- **Train:** 12,285 images (80.00%, 43/43 classes present)
- **Validation:** 1,537 images (10.01%, 43/43 classes present)
- **Test:** 1,536 images (9.99%, 43/43 classes present)
- **Total Valid Images:** 15,358 images (4 images excluded due to label conflicts).

**Leakage Verification on Repaired Splits:**
- Exact SHA-256 cross-split collisions: **0**
- Perceptual near-duplicate cross-split collisions (dist $\le 2$): **0**

The repaired manifest was saved with full provenance, license tags (CC-BY-4.0), and audit metadata to:
`data_manifests/indian_food_43class_repaired_manifest_v2.json`

---

## Phase 4 — Existing Checkpoint Evaluation & Contamination Proof

### 4.1 Evaluation Against Repaired Splits
The existing checkpoint was evaluated on the repaired splits using standard ImageNet preprocessing (Resize 256, CenterCrop 224, standard normalization).

| Split | Total Samples | Classes Present | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | 95% Wilson CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 12,285 | 43 / 43 | 96.11% | 96.10% | 0.9609 | 0.9610 | [95.76%, 96.43%] |
| **Validation** | 1,537 | 43 / 43 | 89.26% | 89.21% | 0.8925 | 0.8926 | [87.62%, 90.72%] |
| **Repaired Test** | 1,536 | 43 / 43 | 89.91% | 89.87% | 0.8988 | 0.8993 | [88.31%, 91.33%] |

### 4.2 The Contamination Discovery
Why does the checkpoint achieve 89.91% accuracy on the repaired test split when its original test accuracy was only 79.14%?

An audit comparing the image IDs in the repaired test split against the checkpoint's original training set revealed:
- **Repaired Test Set Size:** 1,536 images.
- **Images previously in Original Train Split:** **1,328 images (86.46%)**!
- **Unseen Images in Repaired Test Split:** Only 208 images (13.54%).

Because the original dataset placed 100% of the 21 missing classes into training, any fair split that allocates 10% of those classes to test inevitably draws images that the old model already memorized during its 13 training epochs.

**Conclusion:**  
The 89.91% accuracy figure is artificially inflated by prior training exposure. **The existing checkpoint cannot provide an unbiased evaluation on the repaired test split.** A clean, trustworthy evaluation requires retraining the model from pretrained ImageNet weights using the repaired training and validation splits, followed by one single unbiased evaluation on the repaired test split.

---

## Phase 5 — Deliverables & Automated Verification

### 5.1 Generated Deliverables
1. **Repaired Dataset Manifest:**  
   `data_manifests/indian_food_43class_repaired_manifest_v2.json`
2. **Audit Metrics & Split Summaries:**  
   `outputs/indian_43class_split_audit.json`
3. **Class Sample Counts (Train, Val, Test):**  
   `outputs/indian_43class_class_counts.csv`
4. **Per-Class Precision, Recall, F1, and Support:**  
   `outputs/indian_43class_per_class_metrics.csv`
5. **Full 43x43 Confusion Matrix:**  
   `outputs/indian_43class_confusion_matrix.csv`
6. **Reproducible Evaluation Script:**  
   `src/classification/evaluate_repaired_splits.py`
7. **Automated Test Suite:**  
   `tests/test_repaired_splits_and_evaluation.py`

### 5.2 Verification Suite
All 32 tests in the master test runner (`src/utils/run_unit_tests.py`) passed with 100% success:
- `test_manifest_integrity` [PASS]
- `test_label_mapping_and_checkpoint_compatibility` [PASS]
- `test_class_coverage_across_all_splits` [PASS]
- `test_duplicate_leakage_and_split_disjointness` [PASS]
- `test_perceptual_cross_split_leakage` [PASS]
- `test_metrics_calculation_and_wilson_ci` [PASS]
- `test_checkpoint_contamination_detection` [PASS]

---

## Retraining Plan & Next Actions

To obtain a scientifically valid benchmark:
1. Train MobileNetV3-Small from pretrained ImageNet weights using `data_manifests/indian_food_43class_repaired_manifest_v2.json`.
2. Train strictly on the `train` split (12,285 images).
3. Tune hyperparameters and select checkpoints strictly using the `val` split (1,537 images).
4. Evaluate once on the held-out `test` split (1,536 images) to record the true generalization accuracy.
