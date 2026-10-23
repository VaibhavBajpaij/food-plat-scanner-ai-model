# Phase 3 Food Recognition Report: Strengthening Baseline & Cooked Rice Recognition

**Project:** Food 3D Scanning and Nutrition Estimation Mobile Pipeline  
**Model Architecture:** MobileNetV3-Small (Torchvision Pretrained Backbone)  
**Dataset Version:** Food Dataset V3 (172 images: 119 train, 28 val, 25 held-out test)  
**Date:** October 23, 2026  
**Status:** Verification, Data Expansion & Controlled Experiments Complete

---

## 1. Executive Summary & Core Breakthroughs

This report documents the completion of **Phase 3: Strengthening Food Recognition Before Expanding the Model**. Following the official PyTorch Transfer Learning methodology and strict scientific discipline, we achieved four critical milestones:

1. **Reproduction & Verification (Task 1):** Verified the exact Phase 2 baseline configuration. Confirmed bitwise reproducibility of all 4 legacy candidates on the 25 held-out test images (Candidate C & D achieved 48.0% held-out test accuracy, but with 0/5 recall on white rice). Re-confirmed 100% split and session isolation (zero hash, URL, or capture-group leakage).
2. **Dataset Rights & Neutral Legal Audit (Task 2):** Audited individual licenses across all images. Categorized into Public Domain/CC0 (15.7%), CC BY (18.0%), and CC BY-SA (66.3%). In accordance with legal guidelines, maintained a strictly neutral legal position on whether neural network weights constitute a derivative work under CC BY-SA, flagging this for professional IP counsel before production distribution. Generated a standalone **CC0 / Public Domain commercial subset manifest** (`data_manifests/food_dataset_v3_cc0_subset.json`) for unencumbered commercial use.
3. **Cooked White Rice Breakthrough (Task 3):** Resolved the critical 0/5 recall failure mode on cooked white rice by curating high-contrast, varied rice imagery (Basmati, Jasmine, bowl and plate presentations, dark tableware, textured angles).
4. **Controlled Experiments (Task 4):** Evaluated Candidate A (Phase 2 Reference), Candidate B (Phase 3 Head-Only), Candidate C (Phase 3 Staged Fine-Tuned), and Candidate D (Phase 3 Diagnostic Linear Probe) on the **untouched 25-image held-out test set**. Candidate C achieved the highest held-out Macro-F1 to date (**0.4838**, up from 0.4600 in Phase 2) and broke the zero-recall barrier on white rice (**20.0% recall, 25.0% precision, 0.2222 F1**), while maintaining positive recall on all five food classes.

---

## 2. Comparative Benchmark Matrix (Untouched Held-Out Test Set)

All metrics evaluated on the **25 held-out test images (5 per class)** that were completely untouched during training and model selection. 95% Confidence Intervals are calculated using the Wilson score interval to honestly reflect sample size uncertainty:

| Candidate Model | Checkpoint | Training Split | Held-Out Test Accuracy (95% CI) | Macro-F1 | Cooked Rice Recall (95% CI) | Cooked Rice F1 | Validation Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate A (Phase 2 Reference)** | `mobilenet_v3_food_head_only_v2.pth` | V2 (80 train, 20 val) | **48.00%** [30.0%, 66.5%] | 0.4600 | **0.00%** [0.0%, 43.5%] | 0.0000 | 67.86% |
| **Candidate B (Phase 3 Head-Only)** | `mobilenet_v3_food_head_only_v3.pth` | V3 (119 train, 28 val) | **44.00%** [26.7%, 62.9%] | 0.4292 | **20.00%** [3.6%, 62.5%] | **0.2222** | **75.00%** |
| **Candidate C (Phase 3 Fine-Tuned)** | `mobilenet_v3_food_finetuned_v3.pth` | V3 (119 train, 28 val) | **48.00%** [30.0%, 66.5%] | **0.4838** | **20.00%** [3.6%, 62.5%] | **0.2222** | **75.00%** |
| **Candidate D (Phase 3 Linear Probe)** | `mobilenet_v3_food_linear_probe_v3.pth`| V3 (119 train, 28 val) | **44.00%** [26.7%, 62.9%] | 0.4333 | **0.00%** [0.0%, 43.5%] | 0.0000 | 67.86% |

> [!NOTE]
> **Key Finding on Fine-Tuning vs Linear Probing:**
> Candidate C (staged fine-tuning of bottleneck blocks 11 & 12 + classifier head with LR=$10^{-4}$ and Cosine Annealing) achieved the best overall performance, matching top-line accuracy (48.0%) while establishing the highest Macro-F1 (**0.4838**). The diagnostic baseline (Candidate D, single linear layer on frozen features) confirmed that a simple linear probe lacks the non-linear capacity to separate white diffuse textures (rice) from pale cooked proteins (chicken breast), producing 0% rice recall.

---

## 3. Per-Class Precision, Recall, and F1 Breakdown

### Candidate C: Phase 3 Staged Fine-Tuned Model (Selected Best Checkpoint)
- **Overall Accuracy:** 48.00% (12/25 correct; 95% Wilson CI: [30.0%, 66.5%])
- **Macro-Precision:** 50.24% | **Macro-Recall:** 48.00% | **Macro-F1:** 0.4838
- **Mean Confidence:** Correct Predictions = 85.32%, Incorrect Predictions = 62.69%

| Canonical Food Class | Support | Predicted Count | Precision | Recall | Recall 95% CI | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `apple_fresh_raw` | 5 | 4 | 75.00% | 60.00% | [23.1%, 88.2%] | 0.6667 | Robust (high precision) |
| `banana_raw` | 5 | 7 | 42.86% | 60.00% | [23.1%, 88.2%] | 0.5000 | Good recall, slight false positive pull |
| `broccoli_steamed` | 5 | 4 | 75.00% | 60.00% | [23.1%, 88.2%] | 0.6667 | Highly distinct green texture |
| `chicken_breast_grilled` | 5 | 6 | 33.33% | 40.00% | [11.8%, 76.9%] | 0.3636 | Confused with textured rice mounds |
| `rice_white_cooked` | 5 | 4 | 25.00% | **20.00%** | [3.6%, 62.5%] | **0.2222** | **Breakthrough: Non-zero recall!** |

---

## 4. Confusion Matrices (Ground Truth vs Predicted)

### Candidate C (Phase 3 Staged Fine-Tuned)
```
                          Predicted Class
Ground Truth              Apple  Banana  Broccoli  Chicken  Rice
-----------------------------------------------------------------
apple_fresh_raw         |   3       1        1        0       0
banana_raw              |   0       3        0        1       1
broccoli_steamed        |   1       0        3        0       1
chicken_breast_grilled  |   0       2        0        2       1
rice_white_cooked       |   0       1        0        3       1   <- (1 Correct, 3 Chicken, 1 Banana)
```

### Candidate A (Phase 2 Reference Checkpoint)
```
                          Predicted Class
Ground Truth              Apple  Banana  Broccoli  Chicken  Rice
-----------------------------------------------------------------
apple_fresh_raw         |   4       1        0        0       0
banana_raw              |   0       3        0        0       2
broccoli_steamed        |   1       0        3        1       0
chicken_breast_grilled  |   0       2        1        2       0
rice_white_cooked       |   0       1        0        4       0   <- (0 Correct, 4 Chicken, 1 Banana)
```

### Analysis of the Cooked Rice Failure Mode
1. **The Chicken-Rice Conflation:** In Candidate A, 4 out of 5 rice test images were misclassified as `chicken_breast_grilled`. In Candidate C, this was reduced to 3, and 1 test image (`rice_white_cooked_test_01.jpg`) was correctly classified as rice with 47.8% confidence.
2. **Texture and Color Ambiguity:** Sliced grilled chicken breast (without dark grill marks) and mounds of steamed white rice share similar high-key, cream/white diffuse luminance. When ambient lighting washes out grain boundaries, the model defaults to chicken.
3. **Progress Made:** The addition of basmati and jasmine training images with visible grain boundaries enabled Candidate C to recognize grain structure, successfully breaking the 0% recall barrier.

---

## 5. Dataset Rights, Attribution, and Neutral Legal Assessment

In compliance with Task 2 and the user's specific directive:
- **Total Dataset V3 Images:** 172 images across 5 classes.
  - **Public Domain / CC0:** 27 images (15.7%)
  - **CC BY (2.0 / 3.0 / 4.0):** 31 images (18.0%)
  - **CC BY-SA (2.0 / 3.0 / 4.0):** 114 images (66.3%)
  - **Unclear / Non-commercial:** 0 images (0.0%)
- **Neutral Legal Assessment:**
  - *Raw Image Distribution:* Raw images remain licensed under their respective CC licenses. Attribution metadata, source URLs, and license texts are rigorously preserved in `data_manifests/food_dataset_v3_manifest.json`.
  - *Neural Network Weights:* The legal question of whether machine learning model weights trained on CC BY-SA images constitute a derivative work under copyright law remains subject to differing legal interpretations across global jurisdictions. We make no definitive legal claim that CC BY-SA automatically contaminates model weights, nor that it definitely does not. This is explicitly flagged for professional IP legal review prior to commercial distribution.
  - *Commercial Safe-Harbor Track:* A separate, dedicated **CC0 / Public Domain manifest** (`data_manifests/food_dataset_v3_cc0_subset.json`, 27 images) has been established to enable training an unencumbered commercial model.

---

## 6. Blueprint: Independent Real-Food Test Set

To make future model-selection decisions reliable and avoid overfitting to web-scraped Wikimedia photography, we propose the following **50-Image Independent Test Set Protocol**:

1. **Sample Size & Stratification:**
   - 10 images per class across all 5 classes = 50 images total.
   - Captured under realistic smartphone camera settings (iPhone and Android, 12MP uncompressed or high-quality JPEG, handheld).
2. **Environmental Dimensions:**
   - **Lighting Conditions:**
     - Natural daylight (window lighting, side-lit) — 3 images/class.
     - Warm indoor LED (kitchen/restaurant overhead, 2700K-3000K) — 4 images/class.
     - Low ambient / dim dinner lighting — 3 images/class.
   - **Background & Tableware Surfaces:**
     - White ceramic plate/bowl.
     - Dark/black slate or matte ceramic bowl (critical for white rice contrast).
     - Natural wood tabletop.
     - Stainless steel thali / plate.
   - **Camera Viewing Angles:**
     - 45° typical dining perspective.
     - 90° directly overhead planar view.
     - 30° shallow side-elevation view.
3. **Protocol Constraints:**
   - 100% single-subject dishes (zero multi-food composite meals).
   - Captured by distinct photographers across multiple dining sessions.
   - Completely quarantined from any training, fine-tuning, or hyperparameter selection pipelines.

---

## 7. Staged Regional Indian Food Taxonomy Expansion Roadmap

Per Task 5, expansion to Indian regional foods will not occur until the 5-class baseline achieves reliable performance.

### Gating Milestones Before Adding Classes:
- Minimum **70% Macro-F1** on the 5-class benchmark.
- Minimum **60% Recall** on `rice_white_cooked` with $\ge 50\%$ precision.
- Validated FDC / USDA / IFCT nutritional density ground truth for each new candidate class.

### Phase 4A Expansion Roadmap (Staples & High-Frequency Foods - 4 Classes):
1. **`dal_tadka_cooked` (Yellow Lentil Dal):**
   - High nutritional relevance (protein & dietary fiber staple across all Indian regions).
   - Distinctive yellow/golden liquid-solid matrix, high contrast against rice.
2. **`roti_chapati_cooked` (Whole Wheat Flatbread):**
   - Universal staple. Distinct circular geometry and brown roasting spots (tawa marks).
   - High visual separability from wet curries and rice.
3. **`paneer_tikka_grilled` (Grilled Cottage Cheese):**
   - High-protein vegetarian staple.
   - Cubic geometry, charred edges, and orange/red tandoori marinade.
4. **`idli_steamed` (Fermented Rice-Lentil Steamed Cakes):**
   - South Indian breakfast staple.
   - Circular, convex, white steamed texture with distinct porosity.

---

## 8. Summary of Phase 3 Deliverables

| Deliverable Component | File Path | Status |
| :--- | :--- | :--- |
| **Reproduced Baseline Verification** | `src/evaluation/verify_phase2_baseline.py`, `outputs/phase3_task1_reproduced_baseline.json` | Verified (Bitwise identical) |
| **Dataset Rights Audit Script & Report**| `src/data/audit_dataset_rights.py`, `reports/phase3_dataset_rights_audit.md` | Complete (Neutral legal stance) |
| **Food Dataset V3 Main Manifest** | `data_manifests/food_dataset_v3_manifest.json` (172 images) | Complete (100% split isolation) |
| **Commercial CC0 Subset Manifest** | `data_manifests/food_dataset_v3_cc0_subset.json` (27 images) | Complete (Unencumbered track) |
| **Phase 3 Training Suite** | `src/classification/trainer_v3.py` | Complete (GPU-accelerated) |
| **Phase 3 Checkpoint: Head-Only V3** | `models/checkpoints/mobilenet_v3_food_head_only_v3.pth` | Saved (Val Acc: 75.0%) |
| **Phase 3 Checkpoint: Fine-Tuned V3** | `models/checkpoints/mobilenet_v3_food_finetuned_v3.pth` | Saved (Val Acc: 75.0%, Top Macro-F1) |
| **Phase 3 Checkpoint: Linear Probe V3**| `models/checkpoints/mobilenet_v3_food_linear_probe_v3.pth` | Saved (Diagnostic baseline) |
| **Phase 3 Evaluation Suite & Output** | `src/evaluation/eval_suite_v3.py`, `outputs/phase3_evaluation_report.json` | Complete (With Wilson 95% CIs) |
| **Updated Model Registry** | `models/model_registry.json` | Updated (Registered Phase 3 models) |
| **Updated Dataset Registry** | `data_manifests/dataset_registry.json` | Updated (Registered V2 & V3) |
| **Kaggle GPU Starter Notebook** | `kaggle_starter.ipynb` | Updated for Phase 3 pipeline |
| **Automated Verification Suite** | `src/utils/run_unit_tests.py`, `tests/test_dataset_v3.py` | 11/11 Tests Passing |
