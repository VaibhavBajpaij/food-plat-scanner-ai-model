# Running the Cloud Dataset Acquisition & Training Pipeline on Kaggle (GPU Guide)

This guide provides complete instructions for executing the cloud dataset acquisition, integrity verification, and MobileNetV3-Small transfer learning pipeline on **Kaggle GPU (NVIDIA Tesla T4 / P100)** with **Google Colab** as a secondary backup.

**Source of Truth Repository:** `https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model.git`

---

## 1. Cloud Infrastructure & Resource Sizing

| Resource Dimension | Local Development (Laptop) | Kaggle Cloud GPU (Primary) | Google Colab (Backup) |
| :--- | :--- | :--- | :--- |
| **GPU Accelerator** | NVIDIA GeForce RTX 2050 (4 GB) | **NVIDIA Tesla T4 (16 GB)** or **2x T4** | **NVIDIA T4 / V100 (15–16 GB)** |
| **Storage Quota** | Constrained local SSD | **~20 GB `/kaggle/working` ephemeral disk** | **~78 GB `/content` ephemeral disk** |
| **Network Throughput** | Standard home/office broadband | **Cloud multi-gigabit link (direct to HF/Mendeley)** | **Cloud multi-gigabit link** |
| **Weekly Free Limit** | N/A | **30 GPU hours / week** | Dynamic free tier compute units |
| **Persistence Mechanism** | Permanent local filesystem | **Download `kaggle_run_artifacts.zip` before exit** | **Mount Google Drive (`/content/drive`)** |

### Estimated Sizing & Requirements:
- **Download Storage (Approved Datasets):**
  - Anchor: *Enhanced Indian Food Classification* = **~1.75 GB** (15,404 images in 5 parquet shards).
  - Regional Safe-Harbor: *BD-Food33* = **~350 MB** (1,200 images).
  - Segmentation Ground Truth: *FoodBD* = **~650 MB** (2,800 images).
  - Object Detection: *RASOI* = **~200 MB** (2,000 images).
  - Total Approved Pool = **~2.95 GB compressed** (unpacks to ~4.5 GB raw image cache).
- **GPU VRAM Utilization:**
  - MobileNetV3-Small (Batch Size 32, Image Size 224x224): **~1.8–2.6 GB VRAM peak**.
  - Head-only training (Stage 1) + bottleneck fine-tuning (Stage 2): **100% stable** within the 16 GB VRAM envelope.
- **Compute Time:**
  - 15,404 images across 20 epochs on T4 GPU: **~12–18 minutes total runtime**.

---

## 2. Nine Candidate Datasets: Audit & Compliance Summary

Before acquiring datasets, our automated governance engine classifies all candidates into four strict tiers:

| Candidate Dataset | Licence | Images | Size | Status | Action |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Enhanced Indian Food Classification** | **CC-BY-4.0** | **15,404** | **~1.75 GB** | **Approved Commercial Workflow** | **Anchor acquisition target** |
| **Bangladeshi Food (BD-Food33)** | **CC0 1.0 (Public Domain)** | **1,200** | **~350 MB** | **Approved Commercial Workflow** | **Unencumbered regional safe-harbor** |
| **FoodBD** | **CC BY 4.0** | **2,800** | **~650 MB** | **Approved Commercial Workflow** | **Multi-food segmentation & nutrition** |
| **Indian Spices** | **CC BY 4.0** | **10,991** | **~1.20 GB** | **Approved Commercial Workflow** | **Spice seasoning recognition** |
| **RASOI (Roboflow)** | **CC BY 4.0** | **2,000** | **~200 MB** | **Approved Commercial Workflow** | **Localized dish bounding boxes** |
| **Indian Foods Dataset** | Apache-2.0 / Scraped | ~2,000 | ~450 MB | Requires Review | Curated Apache-2.0 allowed; scrapes blocked |
| **SpiceVision** | Academic | ~3,000 | ~300 MB | Access Not Verified | Excluded until open mirror verified |
| **FoodLens (Google Research)** | Proprietary / Closed | ~30,000 | N/A | Access Not Verified | Closed Google internal dataset |
| **Khana (arXiv:2509.06006)** | **CC BY-NC-ND 4.0** | **131,000** | ~10.0 GB | **Excluded Non-Commercial** | **Strictly quarantined (ComplianceViolationError)** |

---

## 3. Step-by-Step Kaggle Execution Instructions

### A. Environment Initialization
1. Navigate to [Kaggle Notebooks](https://www.kaggle.com/code) and select **New Notebook**.
2. In the right-hand **Notebook Options** panel:
   - **Accelerator**: Select **GPU T4 x1** (or GPU P100).
   - **Internet**: Toggle to **ON** (Mandatory for Git clone and Hugging Face parquet stream).
   - **Persistence**: Select *Variables and Files*.
3. Upload `kaggle_starter.ipynb` from the repository or paste the cells sequentially.

### B. Execution Sequence in `kaggle_starter.ipynb`
- **Cell 1 (Clone Repository):**
  ```python
  !git clone https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model.git /kaggle/working/food-plat-scanner-ai-model
  %cd /kaggle/working/food-plat-scanner-ai-model
  ```
- **Cell 2 (Environment Pre-Flight Audit):**
  Verifies CUDA availability, tests HTTP connectivity to Hugging Face, and checks available `/kaggle/working` disk space.
- **Cell 3 (Install Minimal Dependencies):**
  Installs `pyarrow`, `fastparquet`, and `Pillow`.
- **Cell 4 (Run Resumable Downloader):**
  Executes `EnhancedIndianFoodAdapter.acquire(resume=True)` to stream parquet shards directly to `/kaggle/working/datasets/raw/sohlhealth_indian_food/`.
- **Cell 5 (Image Integrity & Deduplication):**
  Verifies image decodability, checks for corrupted bytes, computes SHA256/dHash, and prints per-class sample counts.
- **Cell 6 (Scientific Unit Tests):**
  Executes `python -m src.utils.run_unit_tests` to verify 18 unit tests across taxonomy, nutrition math, synthetic volume, and pipeline integrity.
- **Cell 7 (Pretrained Model Smoke Test):**
  Runs a 1-epoch / 2-batch forward and backward pass on `models.mobilenet_v3_small(weights=DEFAULT)` to verify CUDA gradient computation and memory headroom.
- **Cell 8 (Execute Staged Transfer Learning):**
  Executes `python -m src.classification.train_kaggle`.
  - Stage 1: Linear probe classification head on frozen backbone.
  - Stage 2: Fine-tuning top bottleneck blocks with cosine annealing.
- **Cell 9 (Artifact Verification & Export):**
  Verifies checkpoint reloadability, measures Wilson 95% confidence intervals, and bundles `mobilenet_v3_food_kaggle_best.pth` and `kaggle_training_report.json` into `/kaggle/working/kaggle_run_artifacts.zip`.

### C. Retrieving Artifacts Before Session Closes
Because `/kaggle/working` is ephemeral:
1. In the right-hand Kaggle panel, open the **Output** tab.
2. Locate `/kaggle/working/kaggle_run_artifacts.zip`.
3. Click the three dots $\to$ select **Download**.
4. Place the downloaded checkpoint into your local repository under `models/checkpoints/` and the report into `outputs/`.

---

## 4. Colab Backup Workflow (`colab_starter.ipynb`)

If Kaggle weekly GPU quotas are exhausted or queued:
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and choose `colab_starter.ipynb`.
3. Set **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
4. In Cell 3, run the Google Drive mount cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Checkpoints and evaluation reports will automatically synchronize to `/content/drive/MyDrive/food_model_checkpoints/` upon completion.
