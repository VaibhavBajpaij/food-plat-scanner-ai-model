# Running the 43-Class Cloud Dataset Acquisition & Training Pipeline on Kaggle (GPU Guide)

This guide provides complete instructions for acquiring the 15,404-image Enhanced Indian Food dataset, verifying image integrity, and fine-tuning the 43-class MobileNetV3-Small transfer learning model on **Kaggle GPU (NVIDIA Tesla T4 / P100)** with **Google Colab** as a secondary backup.

**Source of Truth Repository:** `https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model.git`

---

## 1. Cloud Infrastructure & Sizing (43 Classes, 15,404 Images)

| Resource Dimension | Local Development (Laptop) | Kaggle Cloud GPU (Primary) | Google Colab (Backup) |
| :--- | :--- | :--- | :--- |
| **GPU Accelerator** | NVIDIA GeForce RTX 2050 (4 GB) | **NVIDIA Tesla T4 (16 GB)** or **2x T4** | **NVIDIA T4 / V100 (15–16 GB)** |
| **Storage Quota** | Constrained local SSD | **~20 GB `/kaggle/working` ephemeral disk** | **~78 GB `/content` ephemeral disk** |
| **Network Throughput** | Standard broadband | **Cloud multi-gigabit link (direct to HF/Mendeley)** | **Cloud multi-gigabit link** |
| **Weekly Free Limit** | N/A | **30 GPU hours / week** | Dynamic free tier compute units |
| **Persistence Mechanism** | Permanent local filesystem | **Download `kaggle_indian_43class_artifacts.zip`** | **Mount Google Drive (`/content/drive`)** |

### Sizing & Requirements:
- **Parquet Download Size:** ~1.75 GB across 5 shards (`train_0`, `train_1`, `train_2`, `val_0`, `test_0`).
- **Extracted Images (JPEG):** ~2.50 GB on disk across `/kaggle/working/datasets/processed/indian_food_43class/`.
- **GPU VRAM Utilization (Batch Size 32):** ~1.8–2.6 GB VRAM peak on MobileNetV3-Small (100% stable within Kaggle's 16 GB VRAM envelope).
- **Compute Time:** ~12–18 minutes total runtime for 15 epochs (5 head + 10 bottleneck fine-tuning) on T4 GPU.

---

## 2. Nine Candidate Datasets: Independent Forensic Licence Summary

| Candidate Dataset | Stated Licence | Provenance / Direct Finding | Status |
| :--- | :--- | :--- | :--- |
| **Enhanced Indian Food Classification** | **CC-BY-4.0** | 15,404 photos across 43 Indian dishes from SohlHealth. | **APPROVED (Primary Anchor)** |
| **Bangladeshi Food (BD-Food33)** | **CC0 1.0** | 1,200 photos taken by Daffodil University researchers. Dedicated to Public Domain. | **APPROVED (Safe-Harbor)** |
| **FoodBD** | **CC BY 4.0** | ~2,800 meal images with polygon masks & nutrition estimates. | **APPROVED (Calibration)** |
| **Indian Spices** | **CC BY 4.0** | 10,991 images of 19 whole and ground spices taken under controlled lighting. | **APPROVED (Ingredients)** |
| **RASOI** | **CC BY 4.0** | ~2,000 plated meal photos with bounding boxes from Walchand College. | **APPROVED (Detection)** |
| **Indian Foods Dataset** | Mixed / Scraped | Web scraped community mirrors; original photographer rights unverified. | **REQUIRES RIGHTS REVIEW (Quarantined)** |
| **SpiceVision** | Academic | Conference paper (IEEE NKCon 2024); no open public repository mirror. | **ACCESS NOT VERIFIED (Excluded)** |
| **FoodLens** | Closed / Proprietary | Google Research internal dataset collected from restaurant menus; unpublished. | **ACCESS NOT VERIFIED (Excluded)** |
| **Khana** | **CC BY-NC-ND 4.0** | 131,000+ images. Prohibits commercial monetization and adapted model weights. | **NON-COMMERCIAL (Strictly Quarantined)** |

---

## 3. Step-by-Step Kaggle Execution Sequence

### A. Initialize Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) $\to$ **New Notebook**.
2. Settings panel (right side):
   - **Accelerator**: **GPU T4 x1** (or GPU P100).
   - **Internet**: **ON** (Mandatory for Git clone and Hugging Face parquet stream).
   - **Persistence**: *Variables and Files*.
3. Upload [`kaggle_starter.ipynb`](file:///d:/health%20app/plat%20scaning%20ml%20mode/kaggle_starter.ipynb) or copy cells sequentially.

### B. Cells in `kaggle_starter.ipynb`
- **Cell 1 (Clone Repository):**
  Clones the latest code from `origin/main` into `/kaggle/working/food-plat-scanner-ai-model`.
- **Cell 2 (Environment Pre-Flight Audit):**
  Audits GPU, tests HTTP outbound connectivity to Hugging Face, and checks available disk space in `/kaggle/working` using platform-independent storage metrics.
- **Cell 3 (Install Minimal Dependencies):**
  Runs `pip install -q pyarrow fastparquet Pillow torchvision`.
- **Cell 4 (Run Resumable Downloader):**
  Acquires all 5 Parquet shards (15,404 rows) directly into `/kaggle/working/datasets/raw/sohlhealth_indian_food/`.
- **Cell 5 (Parquet Extraction & Validation):**
  Decodes raw byte structs into JPEGs, computes SHA256 and dHash to drop corrupted files and cross-split near-duplicates, and builds `data_manifests/indian_food_43class_manifest.json`.
- **Cell 6 (Run Unit Tests):**
  Runs `python -m src.utils.run_unit_tests` (all 23 unit and integrity tests pass).
- **Cell 7 (Run 43-Class Smoke Test):**
  Tests `build_mobilenet_v3(num_classes=43)` with dummy batch to verify forward/backward passes and output logits shape `[4, 43]`.
- **Cell 8 (Execute 43-Class Training on 15,404 Images):**
  Runs `train_indian_43class_model(...)`.
  - Stage 1: Linear probe on adapted 43-class head with frozen backbone (5 epochs).
  - Stage 2: Staged fine-tuning of upper bottleneck blocks 10-12 with cosine annealing (10 epochs).
  - Strictly halts with `LegacyDatasetGuardError` if a legacy or small dataset (<1,000 images or <40 classes) is detected.
- **Cell 9 (Artifact Verification & Export):**
  Reloads checkpoint, verifies `num_classes == 43`, and packages `mobilenet_v3_indian_43class_best.pth`, `kaggle_indian_43class_training_report.json`, and `indian_food_43class_manifest.json` into `/kaggle/working/kaggle_indian_43class_artifacts.zip`.

### C. Download Checkpoint Before Session Closes
In the Kaggle right-hand panel:
1. Open the **Output** tab.
2. Locate `/kaggle/working/kaggle_indian_43class_artifacts.zip`.
3. Click three dots $\to$ **Download**.
4. Unpack locally to place the model into `models/checkpoints/` and the report into `outputs/`.
