# Food 3D Scanning and Nutrition Estimation Research Pipeline

An end-to-end, scientifically rigorous research prototype for estimating food identity, segmentation, 3D volume, portion mass, and nutritional composition with analytical uncertainty bounds from smartphone imagery.

---

## Hardware Budget & Target
- **Development Workstation**: Windows 11 with NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)
- **Deployment Target**: Android phones, including constrained 2 GB RAM devices
- **Storage Policy**: Large datasets and virtual environments reside on `D:\` drive

---

## Directory Organization
```
d:\health app\plat scaning ml mode/
├── configs/                  # Runtime, hardware budget, and training YAML configs
├── datasets/                 # Local datasets partitioned by dual-track licensing
│   ├── raw/                  # Downloaded raw images / video frames
│   └── processed/            # Preprocessed and normalized data
├── data_manifests/           # Registries: dataset licenses, unified taxonomy, densities, USDA FDC
│   ├── dataset_registry.json
│   ├── food_taxonomy.json
│   ├── density_registry.json
│   └── fdc_foundation_foods.json
├── models/                   # Model registry and trained weights
│   ├── model_registry.json
│   └── checkpoints/
├── src/                      # Modular Python source code
│   ├── data/                 # Data loaders, taxonomy resolver, sample generator
│   ├── classification/       # MobileNetV3 and DINOv2 classifiers
│   ├── segmentation/         # MobileSAM / FastSAM segmenters
│   ├── reconstruction/       # Feature SfM, depth fusion, and volume calculation
│   ├── nutrition/            # USDA FDC nutrition engine with Gaussian uncertainty
│   ├── evaluation/           # Baseline benchmarks and mobile export
│   └── utils/                # Environment verification and GPU memory management
├── tests/                    # Unit and integration test suite
├── outputs/                  # Benchmark JSON outputs and point clouds
└── reports/                  # Markdown research reports and audit documentation
```

---

## Phase 0 Reproducibility Commands

### 1. Run Environment Audit
```powershell
& ".\.venv\Scripts\python.exe" -m src.utils.verify_env
```

### 2. Run Core Unit Tests
```powershell
& ".\.venv\Scripts\python.exe" -m src.utils.run_unit_tests
```

### 3. Run Phase 0 Baseline Benchmark
```powershell
& ".\.venv\Scripts\python.exe" -m src.evaluation.phase0_benchmark
```

---

## Core Scientific Principles
1. **No False Precision**: Hidden oils, unmeasured salt, and preparation methods are flagged with explicit `UNMEASURED_OR_HIDDEN_FACTOR` states rather than guessing.
2. **Analytical Uncertainty Propagation**: Portion mass uncertainty is modeled via Gaussian propagation:
   $$\sigma_M = \sqrt{V^2 \sigma_\rho^2 + \rho^2 \sigma_V^2}$$
3. **Dual-Track Licensing**: Academic-only datasets (Food-101, Nutrition5k) are strictly quarantined in offline evaluation benchmarks; production models and density registries remain 100% commercial-clean (Apache-2.0 / USDA Public Domain).
