# Comprehensive Dataset Rights, Licensing & Expansion Audit
**Date:** October 2026  
**Scope:** Evaluation of food image datasets for expanding from 172 images to an initial target of 1,000–2,000 unique images for food recognition and segmentation, and migrating training to Kaggle GPU.  
**Repository:** [https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model](https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model)

---

## 1. Executive Summary

Our current dataset (Food Dataset V3) consists of **172 verified real-food images** across 5 classes (`rice_white_cooked`, `chicken_breast_grilled`, `broccoli_steamed`, `apple_fresh_raw`, `banana_raw`), with a strictly isolated 25-image held-out test set and a 27-image CC0 commercial safe-harbor subset. 

To expand our recognition scope towards daily staple foods (prioritizing **cooked rice, dal, roti, dosa, idli, paneer, chicken, vegetables, and common international foods**) to an initial milestone of **1,000–2,000 unique images**, we conducted an exhaustive legal and technical audit of primary academic and open datasets.

### Key Policy Decisions:
1. **Strict Non-Commercial Quarantine:** Popular academic datasets (**Food-101**, **UECFoodPix Complete**, **FoodX-251**) explicitly prohibit commercial model training and redistribution. They are strictly quarantined for offline benchmark comparisons and **excluded from the commercial model training track**.
2. **Dual-Track Data Architecture:**
   - **Track 1: Unencumbered Commercial Safe-Harbor (CC0 / Public Domain):** Zero legal risk, unrestricted worldwide commercial redistribution without attribution requirements.
   - **Track 2: Permissive Open License Track (CC BY 2.0/4.0, Apache 2.0):** Permitted for commercial training and model weights, requiring persistent machine-readable attribution manifests and copyright notices.
   - **Track 3: Copyleft (CC BY-SA) Policy:** Kept segregated with clear provenance. Flagged for neutral legal review before production distribution.
3. **Data Integrity & Anti-Inflation Rules:**
   - Zero tolerance for duplicate images, continuous video frame bursts, or counting synthetic augmentations as real data.
   - Enforce Perceptual Hashing (`dHash` Hamming distance $\le 4$) and SHA256 exact-match deduplication.
   - Strict capture-session / photographer grouping to ensure no session leakage between training, validation, and test splits.
4. **Physical Separation: 2D Classification vs. 3D Calibrated Reconstruction:**
   - 2D food recognition and segmentation classify *what* is on the plate and delineate boundaries.
   - 3D volume and mass recovery require calibrated multi-view stereo / depth sensors, camera pose estimation, and metric scale reference (e.g., plate fiducial). Uncalibrated web photos must never be substituted for calibrated multi-view metric data.

---

## 2. In-Depth Dataset Audit Matrix

| Dataset | Origin / Authors | Raw Size | Original Image Source | Stated License | Commercial Model Training Permitted? | Redistribution Permitted? | Governance Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Food-101** | ETH Zurich (Bossard et al., ECCV 2014) | 101,000 images, 101 classes | Scraped from `foodspotting.com` | Non-Commercial Academic Research Only | **NO** (Strictly Prohibited) | **NO** (Photographer copyrights apply) | **QUARANTINED** (Academic offline benchmark only; excluded from production model weights) |
| **UECFoodPix Complete** | Univ. of Electro-Communications, Tokyo (Okamoto & Yanai, 2021) | 10,000 segmented images, 102 classes | Web crawl (UEC-Food-100/256) | Academic & Educational Use Only | **NO** (Strictly Prohibited without commercial licensing agreement) | **NO** | **QUARANTINED** (Segmentation mIoU reference only) |
| **FoodX-251** | CVPR 2019 FGVC6 Workshop (Kaur et al.) | 158,000 images, 251 fine-grained classes | Extension of Food-101 web crawl | Non-Commercial Research & Educational Use Only | **NO** (Explicitly Prohibited) | **NO** (Explicitly Forbidden) | **QUARANTINED** (Excluded from commercial pipeline) |
| **Google Open Images V7** | Google Research / Flickr | 9M+ images with bounding boxes | Flickr photos with verified Creative Commons | Annotations: CC BY 4.0; Images: CC BY 2.0 / CC BY 4.0 | **YES** (Permitted with attribution) | **YES** (With attribution) | **QUALIFIED FOR TRACK 2** (Selected food classes) |
| **Kaggle Indian Food Images (20 Classes)** | Open community curated / Kaggle | 2,000 images, 20 Indian staple classes | Curated Indian food photography | Apache 2.0 | **YES** (Permitted under Apache 2.0 terms) | **YES** (With Apache 2.0 NOTICE and license text) | **QUALIFIED FOR TRACK 2** (Excellent coverage of Dosa, Idli, Roti, Dal, Paneer) |
| **Kaggle Indian Food (Raw Web Scrapes)** | Various community uploaders | Varies (1k–10k) | Unfiltered Google / Bing image search | "Data files © Original Authors" / Unspecified | **NO / UNCLEAR** | **NO** | **REJECTED** (Unverified web scrapes violate compliance rules) |
| **Wikimedia Commons (Food Categories)** | Wikimedia Foundation contributors | Millions of media files | Verified user uploads with EXIF and structured metadata | CC0, Public Domain, CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0 | **YES** (CC0/PD: Unconditional; CC BY: With attribution; CC BY-SA: Subject to legal neutral stance) | **YES** (Under applicable CC terms) | **PRIMARY QUALIFIED SOURCE** (High-resolution, verified origin, rich metadata) |
| **USDA FoodData Central (FDC)** | USDA Agricultural Research Service | 8,200+ food items | U.S. Government analytical lab measurements | Public Domain (17 U.S.C. § 105) | **YES** (100% Unrestricted worldwide) | **YES** (Unrestricted) | **FOUNDATION METADATA SOURCE** (Ground truth densities, portion weights, macronutrients) |

---

## 3. Separation of 2D Recognition and 3D Metric Reconstruction

A critical architectural failure mode in computer vision food apps is treating collections of 2D still web images as a substitute for 3D multi-view volumetric reconstruction. 

```
+-------------------------------------------------------------------------------------------------------+
|                                    FOOD SCANNER SYSTEM ARCHITECTURE                                   |
+-------------------------------------------------------------------------------------------------------+
                                                    |
         +------------------------------------------+------------------------------------------+
         |                                                                                     |
         v                                                                                     v
+--------------------------------------------------+   +--------------------------------------------------+
|      PIPELINE A: 2D RECOGNITION & SEGMENTATION   |   |   PIPELINE B: 3D METRIC RECONSTRUCTION & MASS    |
+--------------------------------------------------+   +--------------------------------------------------+
| • Scope: Image-level food classification & masks |   | • Scope: Metric scale, 3D mesh, volume & mass    |
| • Input: Uncalibrated RGB still images (Kaggle)  |   | • Input: Multi-view calibrated video / depth     |
| • Targets: MobileNetV3-Small, MobileSAM          |   | • Targets: Monocular depth fusion, SfM / TSDF    |
| • Data Source: 1,000-2,000 audited web images   |   | • Data Source: Calibrated capture with fiducial  |
| • Output: Class logits P(c|I), binary mask M(x,y)|   | • Output: Metric volume V (cm³), Mass M = V·ρ (g)|
| • Milestone: Target of current training pivot    |   | • Milestone: Separate future physical pipeline   |
+--------------------------------------------------+   +--------------------------------------------------+
```

### Why Still Images Cannot Provide 3D Ground Truth:
1. **Scale Ambiguity:** A 2D image of cooked rice in a small 10 cm bowl looks identical to a 30 cm serving platter without metric references.
2. **Missing Volumetric Depth:** Food density varies by aeration and packing (e.g., fluffy steamed basmati rice $\rho \approx 0.65\text{ g/cm}^3$ vs. compressed rice $\rho \approx 0.85\text{ g/cm}^3$). Mass calculation requires:
   $$\text{Mass} = \int_{V} \rho(x, y, z) \, dV \approx V_{\text{mesh}} \times \rho_{\text{canonical}}$$
   which demands accurate metric volume $V_{\text{mesh}}$ derived from calibrated multi-view or metric depth sensors, not 2D bounding boxes.
3. **Strict Boundary:** We strictly decouple 2D recognition training from 3D multi-view calibration.

---

## 4. Target Taxonomy for 1,000–2,000 Image Milestone

To represent real-world dining scenarios with high priority on Indian staples and common global foods, we define an expanded 10-class core taxonomy:

| Canonical ID | English Name | Indian / Regional Names | Category | Target Images | Source Candidates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rice_white_cooked` | Steamed White Rice | Chawal, Bhat, Gohan, Bai Fan | Grain / Staple | 120–150 | Wikimedia (`Category:Rice dishes`), Open Images |
| `dal_cooked` | Cooked Lentil Curry (Dal) | Dal Tadka, Dal Fry, Sambar, Parippu | Legume / Curry | 100–130 | Kaggle Apache 2.0 (`Dal-Curry`), Wikimedia |
| `roti_cooked` | Indian Flatbread (Roti/Chapati) | Roti, Chapati, Phulka, Flatbread | Bread / Staple | 100–130 | Kaggle Apache 2.0 (`Roti`), Wikimedia (`Category:Roti`) |
| `dosa_cooked` | Crisp Fermented Rice Crepe | Dosa, Masala Dosa, Plain Dosa | Breakfast / Fermented | 100–130 | Kaggle Apache 2.0 (`Dosa`), Wikimedia (`Category:Dosa`) |
| `idli_cooked` | Steamed Rice Cake | Idli, Idly, Thatte Idli | Breakfast / Steamed | 100–130 | Kaggle Apache 2.0 (`Idly`), Wikimedia (`Category:Idli`) |
| `paneer_curry` | Cottage Cheese Curry | Palak Paneer, Paneer Butter Masala, Shahi Paneer | Dairy / Protein Curry | 100–130 | Kaggle Apache 2.0 (`Palak Paneer`), Wikimedia |
| `chicken_cooked` | Cooked Chicken (Grilled/Curry) | Murgh Tikka, Butter Chicken, Roast Chicken | Meat / Poultry | 120–150 | Wikimedia (`Category:Chicken dishes`), Open Images |
| `vegetables_cooked` | Cooked Mixed Vegetables | Sabzi, Steamed Broccoli, Stir-fry | Vegetable Dish | 120–150 | Wikimedia (`Category:Vegetable dishes`), Open Images |
| `apple_fresh_raw` | Fresh Apple (Raw) | Seb, Pingguo, Manzana | Fresh Fruit | 100–120 | Wikimedia (`Category:Apples`), Open Images |
| `banana_raw` | Fresh Banana (Raw) | Kela, Xiangjiao, Platano | Fresh Fruit | 100–120 | Wikimedia (`Category:Bananas`), Open Images |
| **Total** | **10 Core Classes** | **Balanced Representation** | | **1,060–1,340 Images** | **100% Permissive / Apache 2.0 / CC** |

---

## 5. Technical Validation Specifications

To ensure scientific integrity and eliminate synthetic or artificial inflation:
1. **Perceptual Hashing & Deduplication:**
   - Compute 64-bit difference hash (`dHash`) using PIL:
     $$\text{HammingDistance}(h_1, h_2) = \sum (h_1 \oplus h_2)$$
   - Any image with Hamming Distance $\le 4$ against an existing image is flagged as a near-duplicate and excluded.
   - Exact SHA256 checksum match triggers instant rejection.
2. **Image Integrity Verification:**
   - Format: Must decode cleanly as JPEG or PNG without decompression bombs or truncated streams.
   - Resolution: Minimum $224 \times 224$ pixels; maximum $4096 \times 4096$ pixels.
   - Channels: 3-channel RGB (auto-convert RGBA/grayscale).
   - Aspect ratio: Clamp between $0.33$ and $3.0$ (reject extreme panoramic banners).
3. **Capture-Session Isolation:**
   - Images captured from the same Flickr album, Wikimedia user upload burst, or video sequence share a `capture_session_id`.
   - All images with the same `capture_session_id` must reside exclusively in ONE split (`train`, `val`, or `test`).
4. **Independent Evaluation Sets:**
   - Historical Test Set (25 images, 5 classes) is retained verbatim for historical baseline tracking.
   - New Independent Expanded Test Set (15–20 images per class across all 10 classes = 150–200 images) is set aside prior to any model tuning.

---

## 6. Estimated Resource Requirements

| Resource Dimension | Specification / Estimate | Limit / Quota | Feasibility Assessment |
| :--- | :--- | :--- | :--- |
| **Dataset Storage** | ~1,200 images $\times$ ~250 KB $\approx$ **300 MB** | Kaggle: 20 GB disk; Local: >100 GB | **PASS** (Minimal footprint) |
| **Local Staging Disk** | ~350 MB (raw images + manifests + caches) | Local workspace disk | **PASS** |
| **Kaggle GPU Accelerator** | NVIDIA T4 (16 GB VRAM) or P100 (16 GB VRAM) | Kaggle Free Tier: 30 hours / week | **PASS** (Extremely lightweight) |
| **MobileNetV3-Small VRAM** | Batch size 32, Image size $224 \times 224$ $\approx$ **~180 MB VRAM** | 16,000 MB VRAM available on T4 | **PASS** (< 2% GPU VRAM used) |
| **Training Duration** | 20 epochs $\times$ ~35 batches $\approx$ 700 steps $\approx$ **~12–18 seconds** on T4 GPU | 30 hours quota | **PASS** (< 0.1% of weekly quota) |
| **Checkpoint Size** | PyTorch state dict $\approx$ **6.2 MB** per checkpoint | Kaggle working output: 20 GB | **PASS** (Instant download and git commit) |
