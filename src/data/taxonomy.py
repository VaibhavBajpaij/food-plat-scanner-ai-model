"""
Unified Food Taxonomy Engine
Provides alias resolution, multilingual canonical mapping, and density queries.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple

class FoodTaxonomyEngine:
    def __init__(self, taxonomy_path: Optional[str] = None, density_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if taxonomy_path is None:
            taxonomy_path = os.path.join(base_dir, "data_manifests", "food_taxonomy.json")
        if density_path is None:
            density_path = os.path.join(base_dir, "data_manifests", "density_registry.json")

        self.taxonomy_path = taxonomy_path
        self.density_path = density_path

        self.classes: Dict[str, Dict[str, Any]] = {}
        self.density_records: Dict[str, Dict[str, Any]] = {}
        self.alias_to_canonical: Dict[str, str] = {}

        self._load_data()

    def _load_data(self):
        if os.path.exists(self.taxonomy_path):
            with open(self.taxonomy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("classes", []):
                    cid = item["canonical_id"]
                    self.classes[cid] = item
                    # Register English name
                    self.alias_to_canonical[item["canonical_name_en"].lower()] = cid
                    # Register regional aliases
                    aliases = item.get("regional_aliases", {})
                    for lang, alias_list in aliases.items():
                        for a in alias_list:
                            self.alias_to_canonical[a.lower().strip()] = cid

        if os.path.exists(self.density_path):
            with open(self.density_path, "r", encoding="utf-8") as f:
                ddata = json.load(f)
                self.density_records = ddata.get("density_records", {})

    def resolve_food(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Resolves a food name or alias into its canonical item.
        Falls back to 'unknown_solid_food' if unclassified.
        """
        q = query.lower().strip()
        if q in self.alias_to_canonical:
            cid = self.alias_to_canonical[q]
            return cid, self.classes.get(cid, {})

        # Partial matching
        for alias, cid in self.alias_to_canonical.items():
            if alias in q or q in alias:
                return cid, self.classes.get(cid, {})

        # Unknown fallback
        fallback_id = "unknown_solid_food"
        return fallback_id, self.classes.get(fallback_id, {
            "canonical_id": fallback_id,
            "canonical_name_en": "Unidentified Food Component",
            "density_ref_id": fallback_id
        })

    def get_density(self, canonical_id: str) -> Tuple[float, float]:
        """
        Returns (mean_density_g_per_cm3, std_density_g_per_cm3)
        """
        rec = self.density_records.get(canonical_id)
        if not rec and canonical_id in self.classes:
            density_ref = self.classes[canonical_id].get("density_ref_id")
            rec = self.density_records.get(density_ref)

        if rec:
            return float(rec["mean_g_per_cm3"]), float(rec["std_g_per_cm3"])
        
        # Default unknown solid food prior
        unk = self.density_records.get("unknown_solid_food", {"mean_g_per_cm3": 0.85, "std_g_per_cm3": 0.20})
        return float(unk["mean_g_per_cm3"]), float(unk["std_g_per_cm3"])
