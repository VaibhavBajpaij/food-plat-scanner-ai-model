# Running the Food 3D Pipeline on Kaggle (Free GPU Guide)

This guide explains how to push your local code repository to **GitHub**, pull it directly into a **Kaggle Notebook**, and run the entire baseline and heavy training workflows using Kaggle's free **NVIDIA Tesla T4 (16 GB VRAM)** GPUs.

---

## Why Kaggle is Ideal for This Project

| Resource | Your Local Laptop | Kaggle Cloud Environment |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 2050 Laptop | **NVIDIA Tesla T4 (or 2x T4)** |
| **GPU VRAM** | **4 GB** | **16 GB (or 32 GB total)** |
| **Free GPU Quota**| Local battery / thermal limits | **30 hours / week free** |
| **Download Speed**| Home Wi-Fi (~20-50 Mbps) | **Cloud Data Center (> 500 Mbps)** |
| **C: Drive Space** | Constrained (~22 GB free) | **~73 GB free disk space per session** |
| **Pre-installed Stack**| Manual setup required | **PyTorch, Torchvision, CUDA, OpenCV pre-installed** |

---

## Step 1: Initialize Git and Push to GitHub (On Your PC)

### A. Stage and Commit the Clean Code Locally
Open PowerShell or your terminal in `D:\health app\plat scaning ml mode`:
```powershell
# 1. Add all project files (your .gitignore automatically protects .venv and heavy weights)
git add .

# 2. Make your initial commit
git commit -m "feat: initial modular food 3d scanning and nutrition pipeline"
```

### B. Create a New Repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Set Repository name to: `food-3d-nutrition-pipeline` (or any name you like).
3. Set visibility to **Public** or **Private**.
4. Leave "Add a README file" **UNCHECKED** (we already created one).
5. Click **Create repository**.

### C. Connect and Push Local Code to GitHub
Copy the commands shown on GitHub, or run:
```powershell
# Rename branch to main
git branch -M main

# Link your local repo to GitHub (replace <YOUR_USERNAME> with your actual GitHub username)
git remote add origin https://github.com/<YOUR_USERNAME>/food-3d-nutrition-pipeline.git

# Push the code
git push -u origin main
```

---

## Step 2: Open a Free Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com/) and sign in.
2. Click the **+ Create** button in the top-left $\to$ Select **New Notebook**.
3. In the right-hand **Notebook Options** panel:
   - **Accelerator**: Change from *None* to **GPU T4 x2** (or GPU T4 x1).
   - **Internet**: Ensure toggle is **ON** (Required for git clone and downloading models).
   - **Persistence**: Select *Variables and Files* (optional).

---

## Step 3: Run the Code in Kaggle (Copy-Paste Cells)

Inside your new Kaggle Notebook, create code cells and run:

### Cell 1: Clone Your Repository from GitHub
```python
# Clone your repository
!git clone https://github.com/<YOUR_USERNAME>/food-3d-nutrition-pipeline.git

# Move into the project directory
%cd food-3d-nutrition-pipeline

# Verify directory contents
!ls -lh
```

### Cell 2: Verify the GPU Environment on Kaggle
```python
# Check GPU hardware allocated by Kaggle (Tesla T4 with 16GB VRAM)
!nvidia-smi

# Run your project's automated environment verification script
import sys
!python -m src.utils.verify_env
```

### Cell 3: Run the Project Unit Tests
```python
# Runs all 6 core tests (taxonomy, density, synthetic volume, Gaussian error propagation)
!python -m src.utils.run_unit_tests
```

### Cell 4: Run the Baseline Food Benchmark
```python
# Executes MobileNetV3-Small on the calibrated plate sample
# Measures latency and peak VRAM on Kaggle's Tesla T4 GPU
!python -m src.evaluation.phase0_benchmark
```

### Cell 5: Inspect Results & Nutrition Predictions Directly in Kaggle
```python
import json

with open("outputs/baseline_results/phase0_baseline_results.json", "r") as f:
    results = json.load(f)

print("Food Identified:", results["nutrition_and_mass"]["canonical_name"])
print("Estimated Mass:", results["nutrition_and_mass"]["estimated_mass_grams"]["mean"], "g")
print("Estimated Calories:", results["nutrition_and_mass"]["estimated_portion_nutrients"]["energy_kcal"]["estimate"], "kcal")
print("95% Confidence Interval for Mass:", results["nutrition_and_mass"]["estimated_mass_grams"]["95_ci"], "g")
```

---

## Step 4: How to Pull Future Updates

Whenever you make improvements on your laptop:
```powershell
# On your PC:
git add .
git commit -m "Update nutrition engine"
git push
```

Inside your Kaggle Notebook:
```python
# On Kaggle:
!git pull
```
Your Kaggle environment will instantly update with your latest local code!
