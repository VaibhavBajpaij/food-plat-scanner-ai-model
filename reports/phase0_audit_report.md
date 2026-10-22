# Phase 0 Audit Report: Environment, Licensing & Baseline Architecture

## Executive Summary
This Phase 0 Audit establishes a reproducible, scientifically honest foundation for the Food 3D Scanning and Nutrition Estimation Research Prototype before commencing any large downloads or training pipelines.

---

## 1. Environment & GPU Audit
- **Operating System**: Windows 11 (AMD64)
- **Host GPU**: NVIDIA GeForce RTX 2050 Laptop GPU
  - **VRAM Total**: 4096 MiB (4.0 GB)
  - **Driver Version**: 617.14
  - **CUDA UMD Version**: 13.4 (Compatible with CUDA 12.x / 11.x runtime wheels)
  - **Active Processes**: 0 (Clean dedicated GPU state)
- **Disk Storage**:
  - `C:\`: 23.77 GB free (Constrained — restricted from model/dataset storage)
  - `D:\`: 85.54 GB free (Designated workspace drive: `D:\health app\plat scaning ml mode\`)
- **Python Runtime**:
  - Isolated Virtual Environment: `D:\health app\plat scaning ml mode\.venv`
  - Python Build: 3.13.9 64-bit (`MSC v.1944 AMD64`)
  - Target PyTorch: `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124`

---

## 2. Model & Dataset Licensing Audit

### Candidate Models
| Model Identifier | Architecture | License | Commercial Permitted | 2 GB Android RAM Compatible | Phase 0 Decision |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **MobileNetV3-Small** | Inverted Residuals + SE (2.5M params) | **Apache-2.0** | **YES** | **YES** (~10 MB FP32, ~2.5 MB INT8) | **ACCEPTED**: Primary lightweight mobile baseline |
| **DINOv2-Small** | ViT-S/14 (21.6M params) | **Apache-2.0** | **YES** | **YES** (~86 MB FP32) | **ACCEPTED**: Feature extractor & zero-shot representation |
| **MobileSAM** | TinyViT Encoder + SAM Decoder (9.8M params) | **Apache-2.0** | **YES** | **YES** (~40 MB) | **ACCEPTED**: Promptable food segmentation |
| **Depth Anything V2 Small** | ViT-Small Depth (24.8M params) | **Apache-2.0** | **YES** | **YES** (~99 MB) | **ACCEPTED**: Monocular depth fusion |
| **FoodSAM** | SAM + MMLab Seg (350M params) | GPL-3.0 / Non-Commercial | **NO** | **NO** (> 1.4 GB) | **REJECTED**: Exceeds commercial rights & mobile RAM limits |

### Candidate Datasets & Registries
| Dataset | Source | Stated License | Commercial Use | Dual-Track Quarantine Status |
| :--- | :--- | :--- | :---: | :--- |
| **USDA FoodData Central** | USDA ARS | **Public Domain (17 U.S.C. § 105)** | **YES** | **TRACK A**: Core commercial-clean nutrition backbone |
| **Empirical Density Database** | USDA / FAO | **Public Domain** | **YES** | **TRACK A**: Core volumetric mass calculation database |
| **Food-101** | ETH Zurich | Academic Non-Commercial / CC BY-NC | **NO** | **TRACK B**: Quarantined strictly for offline research evaluation |
| **Nutrition5k** | Google Research | CC BY-NC 4.0 | **NO** | **TRACK B**: Quarantined strictly for offline benchmark comparison |
| **FoodSeg103** | LARC / CMU / SMU | Academic Research Only | **NO** | **TRACK B**: Quarantined strictly for segmentation mIoU evaluation |

---

## 3. The Core Physical Challenge: Metric Scale & Volume Estimation
As identified, downloading models is trivial compared to solving real-world metric scale ambiguity. In monocular camera geometry:
$$x = K [R \mid t] X$$
Scale is inherently lost without a known baseline, depth sensor, or fiducial.

### Phase 0 Metric Solution
1. **Calibrated Plate Boundary**: The standard dinner plate diameter (nominally 24.0 cm) acts as an ellipse fitting constraint:
   $$s = \frac{d_{\text{plate\_px}}}{24.0\text{ cm}} \implies 1\text{ cm} = \frac{d_{\text{plate\_px}}}{24.0}\text{ px}$$
2. **Reference Marker**: Standard coin (2.4 cm) or card (8.56 cm).
3. **Analytical Mass & Density Error Propagation**:
   $$\sigma_M = \sqrt{V^2 \sigma_\rho^2 + \rho^2 \sigma_V^2}$$
   Treating hidden oils and recipe variance as explicit unknown factors rather than false precision.
