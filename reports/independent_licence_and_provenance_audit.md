# Independent Forensic Audit: Licence Terms, Provenance & Commercial Usability of Food Datasets
**Date:** October 2026  
**Scope:** Independent, primary-source forensic review of image provenance, author copyrights, and license terms across candidate food datasets.  
**Repository:** [https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model](https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model)

---

## 1. Executive Summary

This audit independently investigates the actual origin of the images, photographer copyrights, and legal terms for each dataset, rather than relying on self-reported platform tags. 

Attribution alone does **not** grant commercial rights if the underlying photograph rights were not explicitly licensed under open commercial terms by the copyright holders.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 LEGAL GOVERNANCE MATRIX                                │
├────────────────────────────────┬─────────────────┬─────────────────────────────────────┤
│ Dataset Name                   │ Licence         │ Commercial Workflow Status          │
├────────────────────────────────┼─────────────────┼─────────────────────────────────────┤
│ Enhanced Indian Food (15,404)  │ CC-BY-4.0       │ APPROVED (Commercial Anchor)        │
│ Bangladeshi Food (BD-Food33)   │ CC0 1.0         │ APPROVED (Public Domain Safe-Harbor)│
│ FoodBD (2,800)                 │ CC BY 4.0       │ APPROVED (Commercial Calibration)   │
│ Indian Spices (10,991)         │ CC BY 4.0       │ APPROVED (Commercial Ingredients)   │
│ RASOI (2,000)                  │ CC BY 4.0       │ APPROVED (Commercial Detection)     │
│ Indian Foods (Curated/Scraped) │ Mixed / None    │ REQUIRES RIGHTS REVIEW (Quarantined)│
│ SpiceVision (3,000)            │ Academic Mixed  │ ACCESS UNVERIFIED (Excluded)        │
│ FoodLens (Google Research)     │ Proprietary     │ CLOSED / PROPRIETARY (Excluded)     │
│ Khana (131,000+)               │ CC BY-NC-ND 4.0 │ STRICTLY EXCLUDED (Non-Commercial) │
└────────────────────────────────┴─────────────────┴─────────────────────────────────────┘
```

---

## 2. In-Depth Dataset Forensic Analysis

### 1. Enhanced Indian Food Classification (SohlHealth)
- **Primary Source:** Hugging Face Hub (`SohlHealth/enhanced-indian-food-classification`)
- **Parquet Schema:** 5 shards (`train/0.parquet`, `train/1.parquet`, `train/2.parquet`, `validation/0.parquet`, `test/0.parquet`).
- **Image Feature:** Struct containing binary image bytes (`bytes`) and string path.
- **Label Feature:** String class name covering 43 authentic Indian categories.
- **Image Provenance:** Curated by SohlHealth contributors specifically for food recognition. Images are real-world photographs representing 43 authentic dishes across regional Indian cuisines (North, South, East, West).
- **Licence Terms:** **CC-BY-4.0** (Creative Commons Attribution 4.0 International).
- **Commercial Permissibility:** Section 3(a) of CC-BY-4.0 grants permission to reproduce and Share the Licensed Material, in whole or in part, for Commercial Purposes, and to produce, reproduce, and Share Adapted Material.
- **Obligations:** Maintain machine-readable attribution crediting SohlHealth under CC-BY-4.0.
- **Verdict:** **APPROVED FOR INTENDED COMMERCIAL WORKFLOW**.

---

### 2. Indian Foods Dataset (Curated vs Scraped Web Mirrors)
- **Primary Source:** Kaggle / GitHub (`kanishk307/indian-food-images-dataset`, `nehaprabhavalkar/indian-food-101`)
- **Image Provenance:** A significant portion of community mirrors was scraped from Google and Bing image searches without filtering for Creative Commons licenses. The Kaggle page explicitly notes: *"Data files © Original Authors"*.
- **Licence Terms:** Unstated / All Rights Reserved by individual photographers, despite a few GitHub forks claiming Apache-2.0.
- **Legal Risk:** Scraping images without photographer consent or open licenses creates substantial copyright liability if incorporated into commercial model weights.
- **Verdict:** **REQUIRES LICENCE OR IMAGE-RIGHTS REVIEW**. Quarantined from the commercial track until per-image licenses are individually verified.

---

### 3. Bangladeshi Food (BD-Food33)
- **Primary Source:** Mendeley Data (DOI: 10.17632/37b7c5x84m.1)
- **Authors:** Kashfia Sharmin, Daffodil International University, Bangladesh.
- **Image Provenance:** 1,200+ original high-resolution photographs taken by the research team using smartphone cameras in domestic and restaurant settings across Dhaka, Bangladesh.
- **Licence Terms:** **CC0 1.0 Universal (Public Domain Dedication)**.
- **Commercial Permissibility:** Complete dedication to the public domain. The authors surrendered all copyright and related rights worldwide.
- **Verdict:** **APPROVED FOR INTENDED COMMERCIAL WORKFLOW** (Zero legal encumbrance).

---

### 4. FoodBD (Bangladeshi Meals & Nutrition)
- **Primary Source:** Mendeley Data (DOI: 10.17632/dpm2v4mv29.1)
- **Authors:** Farzana Ahmed et al., United International University, Bangladesh.
- **Image Provenance:** ~2,800 smartphone photographs of plated meals with polygon segmentation masks and nutritionist-calculated macronutrient breakdowns.
- **Licence Terms:** **CC BY 4.0**.
- **Commercial Permissibility:** Permitted for commercial training and model derivation with attribution.
- **Verdict:** **APPROVED FOR INTENDED COMMERCIAL WORKFLOW**.

---

### 5. SpiceVision
- **Primary Source:** IEEE NKCon 2024 academic publication (*"SpiceVision: A Dual-Attention EfficientNetB3 Framework with Multi-Scale CBAM Feature Fusion for Fine-Grained Spice Image Classification"*).
- **Image Provenance:** Collected by academic authors for evaluation of a neural network paper. No centralized open mirror (Zenodo, Hugging Face, or Mendeley) was published.
- **Licence Terms:** Academic publication fair-use terms; underlying image copyrights remain unverified.
- **Verdict:** **DOWNLOAD ACCESS NOT YET VERIFIED / REQUIRES REVIEW**. Excluded from bulk acquisition.

---

### 6. Indian Spices Image Dataset
- **Primary Source:** Mendeley Data (DOI: 10.17632/s977p6h659.1) / Zenodo (DOI: 10.5281/zenodo.10058223).
- **Authors:** Sandip Thite, Kailas Patil et al.
- **Image Provenance:** 10,991 images across 19 categories of Indian whole and ground spices captured under controlled laboratory lighting by the authors.
- **Licence Terms:** **CC BY 4.0**.
- **Verdict:** **APPROVED FOR INTENDED COMMERCIAL WORKFLOW**.

---

### 7. FoodLens (Google Research 2025)
- **Primary Source:** Google Research publication (*"FoodLens: Fine-Grained and Multi-Label Classification of Indian Food Images"*, 2025).
- **Image Provenance:** Collected internally by Google from restaurant menus across India.
- **Licence Terms:** Closed / Proprietary to Google LLC.
- **Verdict:** **DOWNLOAD ACCESS NOT YET VERIFIED (CLOSED / PROPRIETARY)**. Excluded from acquisition.

---

### 8. Khana (A Comprehensive Indian Cuisine Benchmark)
- **Primary Source:** arXiv:2509.06006 / `khana.omkar.xyz`
- **Authors:** Omkar et al.
- **Image Provenance:** 131,000+ images across 80 Indian dishes, standardized to 500x500 px.
- **Stated Licence:** **CC BY-NC-ND 4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0).
- **Legal Analysis:**
  - `NonCommercial (NC)`: Prohibits use of the material for commercial advantage or monetary compensation.
  - `NoDerivatives (ND)`: Prohibits the distribution of any modified material (including derived models if weights constitute an adapted work).
  - Even under legal interpretations where training is treated as computational fair use, deploying a commercial Android application derived from an NC-ND dataset exposes the company to cease-and-desist or commercial infringement litigation.
- **Verdict:** **NON-COMMERCIAL; STRICTLY EXCLUDED FROM COMMERCIAL TRACK**. Quarantined with an automated `ComplianceViolationError` guardrail.

---

### 9. RASOI (Indian Food Object Detection)
- **Primary Source:** Roboflow Universe (`wce-6xgpk/rasoi`)
- **Authors:** Walchand College of Engineering contributors.
- **Image Provenance:** ~2,000 images with localized bounding box annotations for plated Indian meals.
- **Licence Terms:** **CC BY 4.0**.
- **Verdict:** **APPROVED FOR INTENDED COMMERCIAL WORKFLOW**.
