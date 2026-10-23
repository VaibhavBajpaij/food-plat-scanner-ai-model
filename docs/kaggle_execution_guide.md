# Running the Food Recognition & 3D Plate Scanning Pipeline on Kaggle (GPU Guide)

This guide explains how to pull your repository (`https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model`) into a **Kaggle Notebook**, train the food-specific transfer learning models, and execute the comparative evaluation suite using Kaggle's free **NVIDIA Tesla T4 (16 GB VRAM)** GPUs.

---

## 1. Why Run on Kaggle?

| Resource | Local Development (Laptop) | Kaggle Cloud Environment |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 2050 Laptop | **NVIDIA Tesla T4 (16 GB)** or **2x T4 (32 GB)** |
| **GPU VRAM** | **4 GB** | **16 GB** |
| **Free Quota** | Battery & thermal limits | **30 hours per week free** |
| **Pre-installed Stack** | Local Python 3.13 + Torch 2.6 | **PyTorch, Torchvision, CUDA, OpenCV pre-installed** |
| **Disk Space** | Constrained C: drive | **~73 GB ephemeral disk space per session** |

---

## 2. Opening the Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com/) and sign in.
2. Click **+ Create** $\to$ Select **New Notebook**.
3. In the right-hand **Notebook Options** panel:
   - **Accelerator**: Select **GPU T4 x1** (or GPU T4 x2).
   - **Internet**: Toggle to **ON** (Required for git clone and torchvision pretrained weight downloads).
   - **Persistence**: Select *Variables and Files* (optional).

---

## 3. Execution Commands on Kaggle

Inside your Kaggle Notebook, run the following cells:

### Cell 1: Clone the GitHub Repository
```python
# Clone the verified repository
!git clone https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model.git

# Navigate into the project root
%cd food-plat-scanner-ai-model

# Inspect repository tree
!ls -lh
```

### Cell 2: Verify GPU Hardware & Environment
```python
# Check Kaggle Tesla T4 GPU
!nvidia-smi

# Run automated environment verification
import sys
!python -m src.utils.verify_env
```

### Cell 3: Execute Scientific Unit & Integrity Tests
```python
# Runs taxonomy, nutrition uncertainty, synthetic volume, and dataset integrity checks
!python -m src.utils.run_unit_tests
```

### Cell 4: Audit Dataset V2 & Split Isolation
```python
import json

with open("data_manifests/food_dataset_v2_manifest.json", "r", encoding="utf-8") as f:
    m = json.load(f)

print(f"Dataset: {m['dataset_name']} (v{m['version']})")
print(f"Total Images: {m['split_summary']['total']} (Train: {m['split_summary']['train']}, Val: {m['split_summary']['val']}, Test: {m['split_summary']['test']})")

tr_shas = {r['sha256'] for r in m['images'] if r['split'] == 'train'}
va_shas = {r['sha256'] for r in m['images'] if r['split'] == 'val'}
te_shas = {r['sha256'] for r in m['images'] if r['split'] == 'test'}

assert len(tr_shas & va_shas) == 0, "Train-Val collision!"
assert len(tr_shas & te_shas) == 0, "Train-Test collision!"
assert len(va_shas & te_shas) == 0, "Val-Test collision!"
print("Split Isolation Status: 100% PASS (Zero cross-split hash or session leakage)")
```

### Cell 5: Train Controlled Transfer Learning Models (Candidates C & D)
```python
# Trains Candidate C (Head-Only with Data Augmentation) and Candidate D (Staged Fine-Tuning)
!python -m src.classification.trainer_v2
```

### Cell 6: Run Comprehensive 4-Candidate Comparative Evaluation
```python
# Benchmarks Candidates A, B, C, D on the untouched held-out test split
!python -m src.evaluation.eval_suite_v2
```

### Cell 7: Display Comparative Evaluation Matrix
```python
import json

with open("outputs/evaluation_results_v2.json", "r", encoding="utf-8") as f:
    res = json.load(f)

print("=" * 85)
print("HELD-OUT TEST SET EVALUATION MATRIX (UNTOUCHED DATA, 5 SAMPLES PER CLASS)")
print("=" * 85)
print(f"{'Candidate Model':<36} {'Test Acc':<12} {'Macro-F1':<12} {'Correct Conf':<14} {'Tableware Bias'}")
print("-" * 85)
for mname, mdata in res["held_out_test_metrics"].items():
    acc = mdata["accuracy"]
    f1 = mdata["macro_f1"]
    conf = mdata["mean_confidence_correct"]
    kw = mdata.get("kitchenware_bias_count", 0)
    print(f"{mname:<36} {acc:>8.1%}    {f1:>8.3f}       {conf:>6.2f}         {kw} hits")
print("=" * 85)
```

---

## 4. How to Sync Updates Between Local Machine and Kaggle

1. **Commit and push locally**:
   ```powershell
   git add .
   git commit -m "feat: your descriptive commit message"
   git push origin main
   ```
2. **Pull inside Kaggle**:
   ```python
   !git pull origin main
   ```
