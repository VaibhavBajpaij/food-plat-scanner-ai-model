"""
Candidate Dataset Licensing & Rights Auditor
Audits popular food datasets (Food-101, UECFoodPix Complete, FoodX-251,
Open Images V7, Kaggle Indian Food datasets, Wikimedia Commons) for
commercial model training permissibility, redistribution restrictions,
and safe-harbor eligibility.
"""

import os
import sys
import json
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANDIDATE_DATASETS: List[Dict[str, Any]] = [
    {
        "id": "food_101",
        "name": "Food-101 Benchmark",
        "institution": "ETH Zurich (Bossard et al., ECCV 2014)",
        "source_type": "Web scraped from foodspotting.com",
        "license_stated": "Academic Research & Non-Commercial Use Only",
        "commercial_use_permitted": False,
        "redistribution_permitted": False,
        "attribution_required": True,
        "status": "QUARANTINED",
        "notes": "Original photographer copyrights apply. Commercial training and weights redistribution are strictly prohibited."
    },
    {
        "id": "uec_foodpix_complete",
        "name": "UECFoodPix Complete",
        "institution": "Univ. of Electro-Communications, Tokyo (Okamoto & Yanai, 2021)",
        "source_type": "Curated & manually segmented web food images (UEC-Food-100/256)",
        "license_stated": "Academic Research & Educational Use Only",
        "commercial_use_permitted": False,
        "redistribution_permitted": False,
        "attribution_required": True,
        "status": "QUARANTINED",
        "notes": "Academic terms strictly prohibit commercial products, monetization, or internal corporate R&D without explicit licensing agreement."
    },
    {
        "id": "foodx_251",
        "name": "FoodX-251 Fine-Grained Benchmark",
        "institution": "CVPR 2019 FGVC6 Workshop (Kaur et al.)",
        "source_type": "Web image crawl extending Food-101 classes",
        "license_stated": "Non-commercial research and educational purposes only",
        "commercial_use_permitted": False,
        "redistribution_permitted": False,
        "attribution_required": True,
        "status": "QUARANTINED",
        "notes": "Terms explicitly state: 'You will use the data only for non-commercial research and educational purposes. You are not permitted to redistribute the images.'"
    },
    {
        "id": "open_images_v7_food",
        "name": "Google Open Images V7 (Food Subset)",
        "institution": "Google Research / Flickr",
        "source_type": "Flickr CC-licensed photos with verified bounding boxes",
        "license_stated": "Annotations CC BY 4.0; Underlying images CC BY 2.0 / CC BY 4.0",
        "commercial_use_permitted": True,
        "redistribution_permitted": True,
        "attribution_required": True,
        "status": "QUALIFIED_COMMERCIAL_TRACK",
        "notes": "Permitted for commercial training provided photographer attribution and CC license notices are maintained in manifests."
    },
    {
        "id": "kaggle_indian_food_apache2",
        "name": "Kaggle Indian Food Images (20 Classes)",
        "institution": "Curated Open Dataset / Kaggle",
        "source_type": "Curated Indian food photography across 20 staples",
        "license_stated": "Apache 2.0",
        "commercial_use_permitted": True,
        "redistribution_permitted": True,
        "attribution_required": True,
        "status": "QUALIFIED_COMMERCIAL_TRACK",
        "notes": "Permits commercial use, modification, and redistribution. Requires Apache 2.0 license notice and attribution."
    },
    {
        "id": "kaggle_indian_food_unverified",
        "name": "Kaggle Indian Food (Unverified Scrapes)",
        "institution": "Various Community Uploads",
        "source_type": "Unfiltered web crawler search results",
        "license_stated": "Data files © Original Authors / Unspecified",
        "commercial_use_permitted": False,
        "redistribution_permitted": False,
        "attribution_required": False,
        "status": "REJECTED_UNVERIFIED",
        "notes": "Mandatory exclusion under project guidelines: rights are unclear, web scraping without provenance is prohibited."
    },
    {
        "id": "wikimedia_commons_food",
        "name": "Wikimedia Commons Curated Food Media",
        "institution": "Wikimedia Foundation & Community",
        "source_type": "Direct uploads with structured EXIF, author, and license metadata",
        "license_stated": "Public Domain, CC0 1.0, CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0",
        "commercial_use_permitted": True,
        "redistribution_permitted": True,
        "attribution_required": True,
        "status": "PRIMARY_SAFE_HARBOR",
        "notes": "Provides high-resolution real food photography with verified provenance. Separates CC0 pure commercial safe-harbor from CC-BY attribution track."
    },
    {
        "id": "usda_fdc",
        "name": "USDA FoodData Central Foundation Foods",
        "institution": "USDA Agricultural Research Service",
        "source_type": "Official U.S. Federal Government Food Composition Database",
        "license_stated": "Public Domain (17 U.S.C. § 105)",
        "commercial_use_permitted": True,
        "redistribution_permitted": True,
        "attribution_required": False,
        "status": "PRIMARY_SAFE_HARBOR",
        "notes": "Official public domain ground truth for food densities, serving portions, and macronutrients."
    }
]

def run_audit() -> Dict[str, Any]:
    print("=" * 80)
    print("AUDITING CANDIDATE FOOD DATASETS FOR COMPLIANCE & COMMERCIAL VIABILITY")
    print("=" * 80)

    summary = {
        "total_audited": len(CANDIDATE_DATASETS),
        "quarantined_academic": 0,
        "rejected_unverified": 0,
        "qualified_commercial_track": 0,
        "primary_safe_harbor": 0,
        "datasets": []
    }

    for d in CANDIDATE_DATASETS:
        status = d["status"]
        if status == "QUARANTINED":
            summary["quarantined_academic"] += 1
            badge = "[QUARANTINED - ACADEMIC ONLY]"
        elif status == "REJECTED_UNVERIFIED":
            summary["rejected_unverified"] += 1
            badge = "[REJECTED - UNCLEAR RIGHTS]"
        elif status == "QUALIFIED_COMMERCIAL_TRACK":
            summary["qualified_commercial_track"] += 1
            badge = "[QUALIFIED - PERMISSIVE / APACHE 2.0]"
        elif status == "PRIMARY_SAFE_HARBOR":
            summary["primary_safe_harbor"] += 1
            badge = "[SAFE HARBOR - CC0 / PUBLIC DOMAIN / CC-BY]"
        else:
            badge = "[UNKNOWN]"

        print(f"\n* {d['name']} ({d['institution']})")
        print(f"  Status: {badge}")
        print(f"  License: {d['license_stated']}")
        print(f"  Commercial Use: {'YES' if d['commercial_use_permitted'] else 'NO (PROHIBITED)'}")
        print(f"  Redistribution: {'YES' if d['redistribution_permitted'] else 'NO (PROHIBITED)'}")
        print(f"  Assessment: {d['notes']}")

        summary["datasets"].append(d)

    print("\n" + "=" * 80)
    print(f"AUDIT SUMMARY: {summary['total_audited']} datasets analyzed.")
    print(f"  • Safe Harbor / Qualified Commercial: {summary['primary_safe_harbor'] + summary['qualified_commercial_track']}")
    print(f"  • Quarantined Non-Commercial:        {summary['quarantined_academic']}")
    print(f"  • Rejected Unverified / Web Scrapes:  {summary['rejected_unverified']}")
    print("=" * 80)

    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "outputs", "candidate_datasets_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Audit report saved to: {out_path}")

    return summary

if __name__ == "__main__":
    run_audit()
