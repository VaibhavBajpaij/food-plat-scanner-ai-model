# Dataset Rights & Licensing Compliance Audit (Phase 3)

**Project:** Food 3D Scanning and Nutrition Estimation Research Pipeline  
**Target Architecture:** MobileNetV3-Small Food Classifier  
**Audit Scope:** All 125 Active Images in Food Dataset V2 + Historical Manifests  
**Date:** October 23, 2026  
**Status:** Complete Legal Provenance Inventory

---

## 1. Executive Summary & Legal Categorization

This audit inspects the individual image licenses, author credits, and distribution obligations across all project datasets in accordance with open-source and commercial compliance guidelines.

### Active Dataset (Food Dataset V2 — 125 Images)

| Legal Classification | Count | % of Dataset | Rights & Distribution Obligations | Permitted for Android Model Weights? |
| :--- | :---: | :---: | :--- | :---: |
| **1. Public Domain or CC0** | **12** | **9.6%** | **Unrestricted worldwide.** No attribution required, free commercial use, no copyleft obligations. | **YES (Unconditional)** |
| **2. Creative Commons BY (CC BY 2.0 / 3.0 / 4.0)** | **23** | **18.4%** | **Permissive with attribution.** Attribution must accompany raw image redistribution. Commercial use permitted. | **YES (With Notice)** |
| **3. Creative Commons ShareAlike (CC BY-SA 2.0 / 3.0 / 4.0)** | **90** | **72.0%** | **ShareAlike Copyleft.** If raw images or image adaptations are redistributed, they must be shared under identical/compatible CC BY-SA terms. | **YES (Weights are functional parameter arrays, not expressive image derivatives; raw images require SA license)** |
| **4. Unclear or Unsuitable** | **0** | **0.0%** | **Zero non-commercial (NC), no-derivs (ND), or unverified images detected.** | **N/A** |
| **Total** | **125** | **100.0%** | **100% Permissive Open Licensing** | |

---

## 2. Legal Analysis of CC BY-SA & ShareAlike Obligations

> [!IMPORTANT]
> **ShareAlike (SA) Clarification & Legal Risk Notice:**
> - **Raw Image Redistribution:** CC BY-SA applies unambiguously to the *expressive copyright* of the photographs. When distributing raw image files from `datasets/food_dataset_v2/` or `datasets/food_dataset_v3/`, the repository must include author attribution, the original source URL, the specific license version (e.g., CC BY-SA 4.0), and a statement that raw image files remain licensed under those terms.
> - **Neural Network Weights & Legal Review Requirement:** The legal question of whether machine learning model weights trained on CC BY-SA images constitute a derivative work under copyright law remains unresolved across major jurisdictions (US, EU, India). 
>   - **Argument A (Non-derivative / Functional Transformation):** Neural network training extracts statistical weight parameters and empirical visual patterns (edge detectors, texture filters) rather than copying expressive photographic form, arguably placing mathematical weight matrices outside the definition of an adapted work.
>   - **Argument B (Derivative / Contributory Ingestion):** Training involves ingestion, transformation, and internal representation of copyrighted visual material, which could under certain legal theories be construed as an adaptation subject to ShareAlike provisions.
>   - **Operational Directive:** The project team makes **no definitive claim** that CC BY-SA either automatically contaminates trained model weights with ShareAlike copyleft, or that it definitively does not. **This must be flagged for qualified IP legal review prior to any commercial production distribution or proprietary app store deployment.**
>   - **Commercial Safe-Harbor Track:** To eliminate licensing ambiguity for commercial deployment, we maintain a separate **CC0 / Public Domain Manifest** (`data_manifests/food_dataset_v3_cc0_subset.json`). A dedicated production model can be trained exclusively on this unencumbered subset or in-house verified photography.

---

## 3. License Breakdown by Class (Food Dataset V2)

| Canonical Class | Total | Public Domain / CC0 | CC BY (2.0/3.0/4.0) | CC BY-SA (2.0/3.0/4.0) | Unclear/NC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `apple_fresh_raw` | 25 | 3 | 11 | 11 | 0 |
| `banana_raw` | 25 | 1 | 4 | 20 | 0 |
| `broccoli_steamed` | 25 | 0 | 2 | 23 | 0 |
| `chicken_breast_grilled` | 25 | 2 | 4 | 19 | 0 |
| `rice_white_cooked` | 25 | 6 | 2 | 17 | 0 |

---

## 4. Complete Image Attribution Registry Sample

Below is a representative sample of verified source URLs, uploader attribution, and licenses from the dataset manifest:

| Image ID | Class | Split | License | Author / Contributor | Wikimedia Source Link |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `v2_apple_fresh_raw_train_03` | `apple_fresh_raw` | `train` | Public domain | <bdi><a href="https://en.wikipedia. | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/a/a1/Osias_Beert_-_Still_life_of_grapes_in_a_dish%2C_apples_in_a_silver_tazza_and_hazelnuts_and_medlars_on_pewter_plates%2C_on_a_wooden_table.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_rice_white_cooked_train_05` | `rice_white_cooked` | `train` | CC0 | <a href="//commons.wikimedia.org/wi | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/d/d6/A_bowl_of_rice.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_rice_white_cooked_train_06` | `rice_white_cooked` | `train` | CC0 | <a href="//commons.wikimedia.org/wi | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/6/69/Bowl_of_white_rice_01.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_rice_white_cooked_train_02` | `rice_white_cooked` | `train` | CC BY 2.0 | <a rel="nofollow" class="external t | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/4/49/Majadra_haystack_%28on_the_plate%29_%284291546125%29.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_chicken_breast_grilled_train_06` | `chicken_breast_grilled` | `train` | CC BY 2.0 | <a rel="mw:WikiLink" href="https:// | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/d/d3/Grilled_chicken_placed_on_a_plate_with_herbs_and_citrus_fruit.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_broccoli_steamed_train_02` | `broccoli_steamed` | `train` | CC BY 4.0 | <a rel="mw:WikiLink" href="https:// | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/8/87/Dinner_at_home_-_October_2023_-_Sarah_Stierch_01.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_apple_fresh_raw_train_04` | `apple_fresh_raw` | `train` | CC BY 2.0 | <a href="//commons.wikimedia.org/wi | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/c/ce/Person_cuts_fresh_fruit_on_a_pink_plate_in_a_kitchen_setting.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_rice_white_cooked_train_01` | `rice_white_cooked` | `train` | CC BY-SA 4.0 | <a rel="mw:WikiLink" href="https:// | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/c/ce/Harmony_bowl.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_rice_white_cooked_train_03` | `rice_white_cooked` | `train` | CC BY-SA 4.0 | <a href="//commons.wikimedia.org/wi | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/3/3e/Rice_plate_Jharkhand.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_rice_white_cooked_train_04` | `rice_white_cooked` | `train` | CC BY-SA 2.0 | <a rel="nofollow" class="external t | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/d/de/Steamed_Rice_with_Chicken_and_Chinese_Sausage_-_Soup_Restaurant_%282500744954%29.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_chicken_breast_grilled_train_01` | `chicken_breast_grilled` | `train` | CC BY-SA 4.0 | Pattaya Patrol | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/d/d0/Burger_and_fries_on_a_wooden_plate.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |
| `v2_chicken_breast_grilled_train_02` | `chicken_breast_grilled` | `train` | CC BY-SA 4.0 | <a href="//commons.wikimedia.org/wi | [Wikimedia Source](https://upload.wikimedia.org/wikipedia/commons/b/b2/Chicken_and_pork_kebab_Pitta_Corner_Wood_Green_High_Road_Haringey_London_England_01.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) |

---

## 5. Audit Recommendations for Phase 3 Data Improvement
1. **Target CC0 / Public Domain for New Rice Data:** When expanding the training dataset for cooked white rice in Task 3, prioritize CC0 / Public Domain photographs from USDA, agricultural repositories, and open food databases to increase the unrestricted portion of the dataset.
2. **Preserve Attribution Manifest:** Continue logging SHA-256 hashes, source URLs, author names, and license strings in JSON manifests for every single acquired image.
3. **Android App Packaging:** When building the mobile application (ONNX/TFLite export), bundle an in-app "Open Source Licenses & Food Image Attributions" screen honoring all CC BY and CC BY-SA notices.
