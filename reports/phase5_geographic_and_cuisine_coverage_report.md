# Phase 5 — Geographic and Cuisine Coverage Report (Phase 3 Deliverable)
**Taxonomy Version:** 2.0.0 (Global Multi-Regional Food Taxonomy)  
**Date:** October 2026  
**Scope:** Versioned food taxonomy, regional coverage matrix, preparation variants, and gap analysis spanning India and 10 global culinary regions.  
**Repository:** [https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model](https://github.com/VaibhavBajpaij/food-plat-scanner-ai-model)

---

## 1. Governance Principles for Global Food Coverage

1. **Physical Decoupling of Cuisine Origin vs. Capture Location:**
   - *Cuisine Origin:* The cultural, culinary, and geographical heritage of the dish (e.g., `Dosa` originates in South India).
   - *Capture Location:* The actual physical location, country, lighting, and camera hardware where the photograph was taken (e.g., a photo of Dosa taken in a restaurant in London, UK under fluorescent lighting).
   - We strictly record these as two separate metadata attributes. We **never infer capture location from the dish name**.
2. **Anti-Inflation Rule (Representative Dishes $\neq$ Regional Coverage):**
   - Having 50 photos of `Pizza` and `Pasta` does NOT constitute coverage of Europe.
   - Having 100 photos of `Butter Chicken` does NOT constitute coverage of India.
   - We explicitly document unrepresented sub-regions, rural preparations, and tribal cuisines as **KNOWN GAPS**.
3. **Multi-Label vs. Single-Item Meals:**
   - South Asian and global meals often consist of composite platters (Thalis, Bento boxes, Mezze platters).
   - In our taxonomy, composite dishes are tracked with individual sub-component canonical IDs to prevent forcing multi-food scenes into artificial single labels.

---

## 2. Tier 1: Comprehensive Indian Food Coverage Matrix

India represents our primary launch region for the Android food-scanning application. The taxonomy is divided into 10 structured food categories spanning North, South, East, West, and Central India:

| Category | Canonical Class IDs | Representative Dishes & Regional Variants | Cuisine Origin & Regional Hub | Current Local Count (V3) | Target Expansion Count (Phase 4) | Known Variations & Failure Modes |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **1. Rice & Grain Staples** | `rice_white_cooked`, `biryani_hyderabadi`, `biryani_lucknowi`, `pulao_veg`, `khichdi_moong`, `curd_rice` | Steamed white rice (Basmati, Sona Masoori, Ponni), Hyderabadi Dum Biryani, Lucknowi Biryani, Vegetable Pulao, Moong Dal Khichdi, South Indian Curd Rice | Pan-India (South, Deccan, North, East) | 45 | 1,200 | Texture variation (fluffy vs sticky rice), garnish occlusion (fried onions, saffron), bowl vs flat plate presentations. |
| **2. Dal & Lentil Curries** | `dal_tadka`, `dal_makhani`, `dal_fry`, `sambar`, `rasam`, `chole_masala`, `rajma_curry` | Yellow Dal Tadka (Toor/Moong), Dal Makhani (Black Urad/Kidney bean), Udupi Sambar, Tomato Rasam, Punjabi Chole, Kashmiri Rajma | North, South, and Central India | 0 | 1,100 | Gravy consistency (watery rasam vs thick dal makhani), oil/tadka floating on surface, bowl reflections. |
| **3. Breads & Flatbreads** | `roti_chapati`, `naan_butter`, `paratha_aloo`, `paratha_plain`, `puri_deep_fried`, `bhature` | Phulka/Roti (whole wheat), Tandoori Naan (Maida, butter, garlic), Aloo Paratha, Poori (puffed), Bhatura (fermented puffed) | North, Northwest, and Central India | 0 | 1,000 | Puffed vs deflated state, ghee/butter sheen, stacked vs single flatbread, burnt griddle spots. |
| **4. Fermented Breakfasts** | `dosa_masala`, `dosa_plain`, `idli_steamed`, `vada_medu`, `uttapam`, `appam_kerala`, `dhokla_gujarati` | Masala Dosa, Paper Roast, Thatte Idli, Medu Vada (crisp lentil donut), Onion Uttapam, Kerala Appam, Gujarati Khaman Dhokla | South India (Tamil Nadu, Karnataka, Kerala, Andhra), Gujarat | 0 | 1,100 | Folded vs rolled crepe, side condiments (coconut chutney, tomato chutney, sambar) touching main food. |
| **5. Paneer & Dairy Curries** | `paneer_butter_masala`, `palak_paneer`, `shahi_paneer`, `kadai_paneer`, `matar_paneer` | Paneer Butter Masala (creamy tomato-cashew), Palak Paneer (pureed spinach), Shahi Paneer, Kadai Paneer (bell peppers) | North India (Punjab, Mughlai, Delhi) | 0 | 900 | Cream swirls, cube size variations, spinach color shift from bright green to dark olive when cooked. |
| **6. Vegetables & Sabzi** | `broccoli_steamed`, `aloo_gobi`, `bhindi_masala`, `baingan_bharta`, `mixed_veg_sabzi`, `avial_kerala` | Steamed broccoli, Aloo Gobi (potato-cauliflower), Bhindi Masala (okra), Smoked Baingan Bharta, Kerala Avial (coconut yogurt gravy) | Pan-India (North, West, South) | 30 | 1,000 | Dry sabzi vs gravy sabzi, heterogeneous diced components, deep-fried vs sauteed appearance. |
| **7. Poultry, Meat & Seafood** | `chicken_breast_grilled`, `butter_chicken`, `chicken_tikka`, `tandoori_chicken`, `rogan_josh_mutton`, `fish_curry_goan` | Grilled chicken breast, Murgh Makhani, Tandoori chicken (red/orange marination), Kashmiri Rogan Josh, Goan coconut fish curry | Pan-India & Global (Punjab, Kashmir, Goa, Bengal) | 35 | 1,100 | Bone-in vs boneless cuts, heavy sauce concealment, red food coloring in restaurant tandoori. |
| **8. Street Food & Snacks** | `samosa_potato`, `pakora_onion`, `pani_puri`, `pav_bhaji`, `vada_pav`, `kathi_roll` | Punjabi Potato Samosa, Onion Bhaji/Pakoda, Pani Puri / Golgappa, Mumbai Pav Bhaji, Mumbai Vada Pav, Kolkata Kathi Roll | Street food hubs (Mumbai, Delhi, Kolkata) | 0 | 950 | Assembly variation (open pav vs closed bun), tamarind/mint chutney drizzle, crushed sev toppings. |
| **9. Sweets & Desserts** | `gulab_jamun`, `jalebi`, `rasgulla`, `kheer_rice_pudding`, `kulfi_traditional`, `modak` | Gulab Jamun (fried milk balls in sugar syrup), Jalebi (crisp orange spirals), Bengali Rasgulla, Rice Kheer, Pista Kulfi, Maharashtrian Modak | Pan-India (Bengal, Maharashtra, North, South) | 0 | 850 | Sugar syrup glare, silver foil (vark) reflection, garnish of sliced pistachios and almonds. |
| **10. Beverages & Fruits** | `apple_fresh_raw`, `banana_raw`, `chai_masala`, `filter_coffee_south_indian`, `mango_lassi`, `chaas_buttermilk` | Fresh whole apple, ripe yellow banana, Masala Chai, South Indian Filter Coffee, Thick Mango Lassi, Spiced Buttermilk | Pan-India & Global | 62 | 800 | Froth on coffee/tea, glassware vs clay kulhad cups, straw occlusions, fruit ripening stages. |
| **Total Indian & Core** | **43+ Target Classes** | | | **172** | **10,000+** | |

---

## 3. Tier 2: Global Regional Coverage & Expansion Roadmap

Beyond the initial Indian food focus, the application is designed to scale across worldwide cuisines. Below is our formal expansion matrix:

```
+---------------------------------------------------------------------------------------------------------------------+
|                                          GLOBAL CUISINE EXPANSION MATRIX                                            |
+---------------------------------------------------------------------------------------------------------------------+
| Region             | Representative Discovered Classes              | Current Status (V3) | Expansion Source Candidates     |
+--------------------+------------------------------------------------+---------------------+---------------------------------+
| East Asia          | Gohan, Ramen, Gyoza, Sushi, Dim Sum, Kimchi   | 0 images (GAP)      | Wikimedia, Open Images V7       |
| Southeast Asia     | Pad Thai, Pho, Nasi Goreng, Satay, Tom Yum     | 0 images (GAP)      | Wikimedia, Open Images V7       |
| Central/South Asia | Momos, Nihari, Haleem, Kothu Roti, Hilsa Curry | 0 images (GAP)      | SohlHealth, Wikimedia           |
| Europe             | Pasta, Pizza, Paella, Croissant, Bratwurst     | 0 images (GAP)      | Open Images V7, Wikimedia       |
| Middle East        | Hummus, Falafel, Shawarma, Shakshuka, Baklava  | 0 images (GAP)      | Wikimedia, Open Images V7       |
| Africa             | Jollof Rice, Tagine, Couscous, Injera, Bobotie | 0 images (GAP)      | Wikimedia Commons               |
| North America      | Burger, Hot Dog, Fries, Mac & Cheese, Poutine  | 0 images (GAP)      | Open Images V7, Wikimedia       |
| Central/Carib.     | Tacos, Burritos, Guacamole, Jerk Chicken       | 0 images (GAP)      | Open Images V7, Wikimedia       |
| South America      | Empanadas, Ceviche, Feijoada, Arepas, Asado    | 0 images (GAP)      | Open Images V7, Wikimedia       |
| Oceania            | Meat Pie, Pavlova, Barramundi, Lamington       | 0 images (GAP)      | Wikimedia Commons               |
+---------------------------------------------------------------------------------------------------------------------+
```

### Detailed Regional Breakdown:

### 1. East Asia (China, Japan, Korea)
- **Representative Classes:** `rice_steamed_gohan`, `ramen_tonkotsu`, `gyoza_pork`, `sushi_nigiri`, `dim_sum_har_gow`, `kimchi_cabbage`, `bibimbap`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Broth clarity (clear shoyu vs milky tonkotsu), sushi toppings (raw salmon, tuna, tamago), dumpling folds.
- **Known Gaps:** Regional Chinese cuisines (Sichuan, Cantonese, Hunan) are heavily underrepresented in open datasets.

### 2. Southeast Asia (Thailand, Vietnam, Indonesia, Malaysia, Philippines)
- **Representative Classes:** `pad_thai`, `pho_beef`, `nasi_goreng`, `satay_chicken`, `tom_yum_goong`, `adobo_chicken`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Fresh herb garnishes (coriander, mint, Thai basil) that obscure noodles/rice beneath.
- **Known Gaps:** Street vendor presentations (banana leaf wraps, plastic bag packaging) differ drastically from restaurant plating.

### 3. Central & South Asia (Pakistan, Bangladesh, Sri Lanka, Nepal, Afghanistan)
- **Representative Classes:** `momos_steamed`, `nihari_beef`, `haleem`, `kothu_roti_sri_lanka`, `hilsa_fish_curry_bengal`.
- **Current Evidence:** Partially covered by overlapping Indian Mughlai and Bengali dishes in SohlHealth (e.g. `momos`).
- **Preparation Variants:** Heavy spiced bone-marrow stews, string hopper noodles.
- **Known Gaps:** Afghan and Central Asian flatbreads and pilafs (Kabuli Pulao).

### 4. Europe (Italy, France, Spain, Germany, UK, Greece, Eastern Europe)
- **Representative Classes:** `pizza_margherita`, `pasta_carbonara`, `paella_valenciana`, `croissant_butter`, `bratwurst_mustard`, `greek_salad`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Pasta shapes (penne, spaghetti, fusilli, ravioli), crust thickness in pizza (Neapolitan vs Roman).
- **Known Gaps:** Eastern European soups (Borscht, Solyanka) and Scandinavian rye-based smørrebrød.

### 5. The Middle East (Levant, Arabian Gulf, Persia, Turkey)
- **Representative Classes:** `hummus_tahini`, `falafel_crispy`, `shawarma_wrap`, `shakshuka_eggs`, `baklava_pistachio`, `kabab_koobideh`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Dip garnishes (olive oil pools, sumac, whole chickpeas), flatbread scoop techniques.
- **Known Gaps:** Gulf Khaleeji rice dishes (Mandi, Kabsa, Machboos).

### 6. Africa (North, West, East, and Southern Africa)
- **Representative Classes:** `jollof_rice_west_africa`, `tagine_moroccan`, `couscous_steamed`, `injera_ethiopian`, `bobotie_south_african`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Spongy grey-brown injera flatbread with multiple dollops of wats/tibs, jollof redness and smokiness.
- **Known Gaps:** Central and Southern African staple porridge dishes (Sadza, Ugali, Fufu) are almost completely absent from standard vision benchmarks.

### 7. North America (United States, Canada)
- **Representative Classes:** `burger_cheeseburger`, `french_fries`, `macaroni_and_cheese`, `clam_chowder`, `poutine_quebec`.
- **Current Evidence:** Partially covered via global classes (`burger` in SohlHealth).
- **Preparation Variants:** Bun seedings, melted cheese coverage, gravy saturation on fries.
- **Known Gaps:** Regional Southern BBQ styles (Kansas City sweet vs Carolina vinegar).

### 8. Central America & Caribbean
- **Representative Classes:** `tacos_al_pastor`, `burrito_beef`, `guacamole_fresh`, `jerk_chicken_jamaican`, `pupusas_salvadoran`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Corn vs flour tortillas, cilantro/onion relish, plantain side dishes.
- **Known Gaps:** Caribbean stews and indigenous Mesoamerican masa preparations.

### 9. South America
- **Representative Classes:** `empanadas_baked`, `ceviche_peruvian`, `feijoada_black_bean`, `arepas_corn`, `asado_grilled_beef`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Cured raw fish in tiger's milk (ceviche) with red onions and sweet potato.
- **Known Gaps:** Amazonian river fish dishes and Andean potato biodiversity.

### 10. Oceania (Australia, New Zealand, Pacific Islands)
- **Representative Classes:** `meat_pie_australian`, `pavlova_fruit`, `barramundi_grilled`, `lamington_cake`.
- **Current Evidence:** 0 images on disk.
- **Preparation Variants:** Meringue crust with passionfruit/kiwi, pastry crimping.
- **Known Gaps:** Pacific Island earth-oven feasts (Hāngī, Umu) and taro-root preparations.

---

## 4. Gap Analysis & Roadmap to 10,000+ Verified Images

```
+-------------------------------------------------------------------------------------------------------------+
|                                  MILESTONE GAP & PROGRESSION ROADMAP                                        |
+-------------------------------------------------------------------------------------------------------------+
| Dataset Phase      | Verified Images | Classes | Geographic Scope                | Status                   |
+--------------------+-----------------+---------+---------------------------------+--------------------------+
| Phase 1 (Baseline) | 32 images       | 5       | Global Staples                  | Completed (Legacy)       |
| Phase 2            | 125 images      | 5       | Global Staples                  | Completed (Audited)      |
| Phase 3 (Active)   | 172 images      | 5       | Global Staples (Rice Focus)     | Completed (Current Tree) |
| Phase 4 (Milestone)| 15,404+ images  | 43+     | In-Depth Indian + Global Core   | Planned & Implemented    |
| Future Expansion   | 50,000+ images  | 150+    | Worldwide Multi-Regional Dishes | Post-Milestone Track     |
+-------------------------------------------------------------------------------------------------------------+
```

### Actionable Acquisition Strategy for 10,000+ Images:
1. **Primary Anchor (Immediate 15,404 Images):**
   - Download and ingest `SohlHealth/enhanced-indian-food-classification` (CC-BY-4.0).
   - Instant coverage across **43 authentic Indian classes** (biryanis, curries, dals, dosas, naans, samosas, chole, paneer).
   - Exceeds the 10,000-image requirement while maintaining strict open licensing (CC-BY-4.0).
2. **Supplemental Multi-Food & Global Anchor (2,000–3,000 Images):**
   - Ingest Google Open Images V7 food subset and Wikimedia Commons verified categories for international staples and multi-food bounding boxes.
3. **Automated Pipeline Enforcement:**
   - Execute perceptual hash deduplication (`dHash`) and capture-session isolation so that zero duplicate or contaminated images enter the training or test splits.

Proceeding to **Phase 4: Automated Resumable Acquisition Pipeline**.
