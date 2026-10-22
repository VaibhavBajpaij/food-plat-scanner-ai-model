"""
Nutrition and Mass Estimation Engine
Computes portion mass and macronutrient breakdown with analytical Gaussian uncertainty propagation.
"""

import os
import json
import math
from typing import Dict, Any, Optional
from src.data.taxonomy import FoodTaxonomyEngine

class NutritionEstimationEngine:
    def __init__(self, fdc_data_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if fdc_data_path is None:
            fdc_data_path = os.path.join(base_dir, "data_manifests", "fdc_foundation_foods.json")

        self.taxonomy = FoodTaxonomyEngine()
        self.fdc_data: Dict[str, Any] = {}
        if os.path.exists(fdc_data_path):
            with open(fdc_data_path, "r", encoding="utf-8") as f:
                self.fdc_data = json.load(f).get("foods", {})

    def estimate_portion(
        self,
        food_query: str,
        volume_cm3: float,
        volume_std_cm3: float = 0.0,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates mass and nutrients for a given food volume and uncertainty.
        Propagates density and volume variance:
        sigma_M = sqrt(V^2 * sigma_rho^2 + rho^2 * sigma_V^2)
        """
        canonical_id, class_info = self.taxonomy.resolve_food(food_query)
        mean_density, std_density = self.taxonomy.get_density(canonical_id)

        # Default volume uncertainty to 8% if unspecified
        if volume_std_cm3 <= 0.0:
            volume_std_cm3 = max(0.08 * volume_cm3, 1.0)

        # Mass calculation: M = V * rho
        mass_g = volume_cm3 * mean_density
        # Gaussian uncertainty propagation
        mass_variance = (volume_cm3 ** 2) * (std_density ** 2) + (mean_density ** 2) * (volume_std_cm3 ** 2)
        mass_std_g = math.sqrt(mass_variance)

        # 95% confidence interval (+- 1.96 * sigma)
        mass_ci_lower = max(0.0, mass_g - 1.96 * mass_std_g)
        mass_ci_upper = mass_g + 1.96 * mass_std_g

        # Fetch nutrition per 100g
        food_nutr = self.fdc_data.get(canonical_id, self.fdc_data.get("unknown_solid_food", {}))
        nutrients_100g = food_nutr.get("nutrients_per_100g", {})
        prep_assumptions = food_nutr.get("preparation_assumptions", "Standard preparation.")

        portion_nutrients = {}
        multiplier = mass_g / 100.0

        for nut_name, val in nutrients_100g.items():
            if isinstance(val, (int, float)):
                nut_val = round(val * multiplier, 2)
                nut_std = round(val * (mass_std_g / 100.0), 2)
                portion_nutrients[nut_name] = {
                    "estimate": nut_val,
                    "uncertainty_std": nut_std,
                    "95_ci": [round(max(0.0, nut_val - 1.96 * nut_std), 2), round(nut_val + 1.96 * nut_std, 2)],
                    "unit": "kcal" if "energy" in nut_name else ("mg" if "mg" in nut_name else "g")
                }
            else:
                portion_nutrients[nut_name] = {
                    "estimate": val,
                    "status": "UNMEASURED_OR_HIDDEN_FACTOR"
                }

        return {
            "query": food_query,
            "canonical_id": canonical_id,
            "canonical_name": class_info.get("canonical_name_en", "Unidentified Food Component"),
            "classification_confidence": round(confidence, 3),
            "estimated_volume_cm3": {
                "mean": round(volume_cm3, 2),
                "std": round(volume_std_cm3, 2)
            },
            "density_g_per_cm3": {
                "mean": round(mean_density, 3),
                "std": round(std_density, 3)
            },
            "estimated_mass_grams": {
                "mean": round(mass_g, 1),
                "std": round(mass_std_g, 1),
                "95_ci": [round(mass_ci_lower, 1), round(mass_ci_upper, 1)]
            },
            "nutrients_per_100g": nutrients_100g,
            "estimated_portion_nutrients": portion_nutrients,
            "preparation_assumptions": prep_assumptions,
            "uncertainty_model": "Analytical Gaussian propagation: sigma_M = sqrt(V^2*sigma_rho^2 + rho^2*sigma_V^2)"
        }
