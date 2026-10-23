# Phase 5 — Comprehensive Project Audit (Phase 1 Baseline)
**Date:** October 2026  
**Scope:** Complete forensic audit of existing code, models, datasets, checkpoints, evaluation reports, and cloud execution status prior to global dataset expansion and large-scale training.  
**Repository:** [https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model](https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model) (`main` branch)

---

## 1. Executive Summary & Verification Matrix

| Audit Dimension | Verified Fact / Ground Truth | Classification Status |
| :--- | :--- | :--- |
| **Pretrained Vision Backbones** | `MobileNetV3-Small` (ImageNet-1k, Apache-2.0, 2.54M params) is the only active local backbone. `DINOv2-Small`, `MobileSAM`, and `Depth Anything V2 Small` are registered in `models/model_registry.json` but weights are not stored locally. | **VERIFIED FACT** |
| **Local Checkpoints Present** | 7 checkpoint files in `models/checkpoints/` (all 5.94 MB, except linear probe at 3.68 MB). | **VERIFIED FACT** |
| **Kaggle Execution Status** | **Zero checkpoints** have been trained on Kaggle or downloaded back. All existing checkpoints were trained on the local laptop GPU (NVIDIA RTX 2050 4GB). `kaggle_starter.ipynb` and `src/classification/train_kaggle.py` are prepared and committed to Git, but have not yet executed in the cloud. | **VERIFIED FACT** |
| **Local Images on Disk** | 347 total image files across directories, representing exactly **220 unique images** by SHA256 checksum. | **VERIFIED FACT** |
| **Active Dataset V3 Size** | Exactly **172 unique verified images** across 5 classes (`train`: 119, `val`: 28, `test`: 25). Zero cross-split leakage. | **VERIFIED FACT** |
| **Commercial Rights Coverage** | **27 images** verified as Public Domain / CC0 (unencumbered commercial safe-harbor); **31 images** CC BY (commercial with attribution); **114 images** CC BY-SA (segregated with neutral legal disclosure). | **VERIFIED FACT** |
| **Geographic Capture Metadata** | **0 out of 172 images** have verified image capture locations (GPS or camera city/country). All 172 are recorded as `Unknown` for capture location. Cuisine origin is derived from taxonomy, NOT photograph location. | **MISSING INFORMATION / GAP** |
| **Current Best Model Performance** | `mobilenet_v3_food_finetuned_v3.pth` achieves **48.0% Test Accuracy** (12/25) with a 95% Wilson CI of **[30.0%, 66.5%]** and Macro-F1 of **0.4838**. Cooked rice recall broke the 0% barrier to reach **20.0% (1/5)**. | **VERIFIED FACT** |

---

## 2. Checkpoints & Architecture Audit

Every checkpoint in `models/checkpoints/` was inspected via PyTorch:

| Checkpoint File | File Size | Epoch | Reported Val Acc | Architecture & Parameters | Training Dataset | Training Environment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mobilenet_v3_food_finetuned_v3.pth` | 5.94 MB | 1 | **75.0%** | MobileNetV3-Small + Staged Upper Residual Blocks (1,535,127 params) | Dataset V3 (172 images) | Local Laptop GPU (RTX 2050) |
| `mobilenet_v3_food_head_only_v3.pth` | 5.94 MB | 9 | **75.0%** | MobileNetV3-Small + Custom 2-Layer MLP Head (1,535,127 params) | Dataset V3 (172 images) | Local Laptop GPU (RTX 2050) |
| `mobilenet_v3_food_linear_probe_v3.pth` | 3.68 MB | 12 | **67.9%** | MobileNetV3-Small + Single Linear Layer (942,039 params) | Dataset V3 (172 images) | Local Laptop GPU (RTX 2050) |
| `mobilenet_v3_food_head_only_v2.pth` | 5.94 MB | 7 | **65.0%** | MobileNetV3-Small + Custom 2-Layer MLP Head (1,535,127 params) | Dataset V2 (125 images) | Local Laptop GPU (RTX 2050) |
| `mobilenet_v3_food_finetuned_v2.pth` | 5.94 MB | 1 | **60.0%** | MobileNetV3-Small + Fine-Tuned Upper Residuals (1,535,127 params) | Dataset V2 (125 images) | Local Laptop GPU (RTX 2050) |
| `mobilenet_v3_food_head_only.pth` | 5.94 MB | 4 | **50.0%** | MobileNetV3-Small + 5-Class Head (1,535,127 params) | Finetune Dataset (32 images) | Local Laptop GPU (RTX 2050) |
| `mobilenet_v3_food_finetuned.pth` | 5.94 MB | 1 | **40.0%** | MobileNetV3-Small + Fine-Tuned Upper Residuals (1,535,127 params) | Finetune Dataset (32 images) | Local Laptop GPU (RTX 2050) |

### Exact Pretrained Weights Loaded:
- **Source:** PyTorch Torchvision official release `torchvision.models.MobileNet_V3_Small_Weights.DEFAULT`.
- **Pretrained Dataset:** ImageNet-1k (1,281,167 training images across 1,000 object categories).
- **Backbone License:** Apache-2.0.
- **Top-1 Accuracy on ImageNet-1k:** 67.668%.
- **Classification Head Adaptation:**
  - Standard head: `Linear(576, 1024) -> Hardswish() -> Dropout(p=0.2) -> Linear(1024, num_classes)`.
  - Linear probe: `Dropout(p=0.2) -> Linear(576, num_classes)`.

---

## 3. Dataset Audit: Downloaded Local Images vs. Candidate Lists

### A. What is Actually Downloaded Locally on Disk:
A recursive filesystem scan of `datasets/` identified:
- **Total image files found on disk:** 347 files.
- **Total UNIQUE images by SHA256 checksum:** **220 unique images**.
- **Duplicate instances across directories:** 127 duplicate copies (e.g., images reused between Dataset V2 and V3).

Breakdown of local directories:
1. `datasets/food_dataset_v3/`: **172 images** (119 train, 28 val, 25 test).
2. `datasets/food_dataset_v2/`: **125 images** (80 train, 20 val, 25 test).
3. `datasets/finetune/`: **32 images** (22 train, 10 val).
4. `datasets/eval/`: **17 images** (Phase 0 smoke test).
5. `datasets/raw/`: **1 image** (`sample_plate_apple.png`).

### B. What is Merely Discovered or Listed as Candidates (NOT Downloaded):
The following datasets are documented in registries and planning scripts but have **zero local images downloaded**:
1. **FoodLens (Google Research 2025):** 0 images downloaded.
2. **Enhanced Indian Food Classification Dataset (HuggingFace SohlHealth, 15,404 images):** 0 images downloaded.
3. **Google Open Images V7 (Food categories):** 0 images downloaded.
4. **Kaggle Indian Food Images Dataset (Apache 2.0, 20 classes / 2,000 images):** 0 images downloaded.
5. **Food-101 Benchmark (ETH Zurich, 101,000 images):** 0 images downloaded (Quarantined academic).
6. **UECFoodPix Complete (Yanai Lab, 10,000 images):** 0 images downloaded (Quarantined academic).
7. **FoodX-251 (CVPR 2019, 158,000 images):** 0 images downloaded (Quarantined academic).
8. **Candidate Pool V4 (Planning script discovered 977 file titles):** Metadata discovered; bulk image binaries not yet downloaded.

---

## 4. Class Distribution, Commercial Rights & Geographic Gaps

### A. Per-Class Image Counts (Active Dataset V3):
| Canonical ID | Canonical Food Name | Train | Val | Held-Out Test | Total Images |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rice_white_cooked` | Steamed White Rice | 32 | 8 | 5 | **45** |
| `chicken_breast_grilled` | Grilled Chicken Breast | 24 | 6 | 5 | **35** |
| `apple_fresh_raw` | Fresh Apple (Raw) | 24 | 6 | 5 | **35** |
| `broccoli_steamed` | Steamed Broccoli | 21 | 4 | 5 | **30** |
| `banana_raw` | Fresh Banana (Raw) | 18 | 4 | 5 | **27** |
| **Total** | **5 Baseline Classes** | **119** | **28** | **25** | **172** |

### B. Commercial Rights Breakdown:
- **Unencumbered Commercial Safe-Harbor (Public Domain / CC0):** **27 images** (18 train, 8 val, 1 test). Dedicated manifest: `data_manifests/food_dataset_v3_cc0_subset.json`.
- **Permissive Attribution Track (CC BY 2.0 / 3.0 / 4.0):** **31 images**. Permitted for commercial model weights provided machine-readable attribution manifests are maintained.
- **Copyleft (CC BY-SA 2.0 / 3.0 / 4.0):** **114 images**. Kept strictly segregated. Neutral legal policy applied pending formal IP clearance.
- **Unclear / Non-Commercial:** **0 images** (strict zero-tolerance policy enforced).

### C. Geographic Metadata & Missing Information:
- **Image Capture Location:** **UNKNOWN for 100% of local images (172/172)**. Wikimedia Commons contributors uploaded these files from worldwide locations, but camera EXIF GPS tags are stripped or unrecorded.
- **Cuisine Origin:** Derived solely from food taxonomy, NOT from photo location. All 5 current classes represent global staples. No specific regional Indian cuisines (e.g. Chettinad, Mughlai, Punjabi, Udupi, Bengali) exist in the current 172-image dataset.

---

## 5. Current Baseline Evaluation Performance & Failure Modes

Evaluated on the **strictly untouched 25-image held-out test set** (5 samples per class):

```
HELD-OUT TEST SET EVALUATION MATRIX (25 UNTOUCHED IMAGES, 5 SAMPLES PER CLASS)
===================================================================================================
Candidate Model                 Test Acc   95% Wilson CI     Macro-F1   Rice Recall  Rice F1
---------------------------------------------------------------------------------------------------
Candidate A (Phase 2 Ref)        48.0%     [30.0%, 66.5%]     0.4600        0.0%      0.0000
Candidate B (Head-Only V3)       44.0%     [26.7%, 62.9%]     0.4292       20.0%      0.2222
Candidate C (Staged Fine-Tune V3)48.0%     [30.0%, 66.5%]     0.4838       20.0%      0.2222
Candidate D (Linear Probe V3)    44.0%     [26.7%, 62.9%]     0.4333        0.0%      0.0000
===================================================================================================
```

### Confusion Matrix (Candidate C - Best Model):
| True Class \ Predicted | Apple | Banana | Broccoli | Chicken | Rice | Class Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple (Raw)** | **3** | 1 | 1 | 0 | 0 | 60.0% (3/5) |
| **Banana (Raw)** | 0 | **3** | 0 | 1 | 1 | 60.0% (3/5) |
| **Broccoli (Steamed)** | 1 | 0 | **3** | 0 | 1 | 60.0% (3/5) |
| **Chicken Breast (Grilled)** | 0 | 2 | 0 | **2** | 1 | 40.0% (2/5) |
| **Rice White (Cooked)** | 0 | 1 | 0 | **3** | **1** | **20.0% (1/5)** |

### Confirmed Failure Modes:
1. **White Rice vs. Chicken Confusion:** 3 out of 5 white rice test images were misclassified as grilled chicken breast due to pale background and color overlap.
2. **Chicken vs. Banana Confusion:** 2 out of 5 chicken test images were misclassified as banana due to yellow curry/lighting hue.
3. **Statistical Uncertainty:** Because the test set contains only 25 images (5 per class), a single misclassified sample alters class recall by 20 percentage points, resulting in a wide 95% confidence interval of [30.0%, 66.5%]. Expanding to a statistically representative test set is essential before claiming production accuracy.
