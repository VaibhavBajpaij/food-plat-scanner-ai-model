# Technical Architecture: Food 3D Scanning & Nutrition Pipeline

## 1. System Pipeline Overview

```
[Smartphone Images / Video]
          │
          ▼
   [Keyframe Selector] ── (Laplacian Sharpness > 100, Farneback Flow > 2.5)
          │
          ├───► [MobileNetV3-Small / DINOv2-Small] ──► Top-K Food Classes
          │
          ├───► [MobileSAM / FastSAM] ──► Visible Food Mask & Boundaries
          │
          └───► [OpenCV SIFT/ORB + Essential Matrix] ──► Sparse 3D Point Cloud
                        │
                        ▼
                [Depth Anything V2 Small] ──► Dense Monocular Depth Fusion
                        │
                        ▼
              [Metric Scale Calibrator] ── (Known plate diameter 24cm or fiducial)
                        │
                        ▼
               [3D Volume Estimator] ──► V ± σ_V (cm³)
                        │
                        ▼
               [Taxonomy & Density Engine] ──► ρ ± σ_ρ (g/cm³)
                        │
                        ▼
              [Analytical Mass Estimator] ──► M = V · ρ ± σ_M (g)
                        │
                        ▼
             [USDA FoodData Central Engine] ──► Nutrients per Portion (95% CI)
```

## 2. Mathematical Modeling of Uncertainty

### 2.1 Volume Estimation
When extracting 3D geometry from a multi-view point cloud:
1. Estimate plane parameters of the plate: $Ax + By + Cz + D = 0$ using RANSAC.
2. Filter points belonging to food cluster $k$ located strictly above the plate plane.
3. Compute convex hull / Poisson mesh volume $V_k$ in camera units.
4. Scale by metric factor $s$ (px/cm or camera units to cm):
   $$V_{\text{metric}} = \frac{V_{\text{raw}}}{s^3}$$
   The standard deviation $\sigma_V$ reflects point noise and calibration variance:
   $$\sigma_V \approx \sqrt{\left(\frac{1}{s^3}\right)^2 \sigma_{V_{\text{raw}}}^2 + \left(\frac{3 V_{\text{raw}}}{s^4}\right)^2 \sigma_s^2}$$

### 2.2 Mass Calculation
$$M = V \cdot \rho$$
By Taylor expansion first-order error propagation for uncorrelated variables $V$ and $\rho$:
$$\sigma_M = \sqrt{V^2 \sigma_\rho^2 + \rho^2 \sigma_V^2}$$

### 2.3 Nutritional Portion Scaling
For nutrient $N$ with concentration $c_N$ (units per 100 g):
$$N_{\text{portion}} = c_N \cdot \frac{M}{100}$$
$$\sigma_{N_{\text{portion}}} = c_N \cdot \frac{\sigma_M}{100}$$
95% Confidence Interval:
$$[N_{\text{lower}}, N_{\text{upper}}] = [\max(0, N_{\text{portion}} - 1.96 \sigma_{N_{\text{portion}}}), N_{\text{portion}} + 1.96 \sigma_{N_{\text{portion}}}]$$

## 3. Hardware Budget Controls (RTX 2050 - 4 GB VRAM)
- Model execution: sequential forward passes rather than concurrent batching.
- Mixed precision: FP16 autocasting via `torch.amp.autocast('cuda')`.
- Peak memory targets:
  - MobileNetV3-Small: < 150 MB
  - DINOv2-Small: < 600 MB
  - MobileSAM: < 700 MB
  - Total active VRAM footprint remains under 1.5 GB, leaving 2.5 GB safety buffer.
