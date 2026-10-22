"""
Tests for Nutrition and Mass Estimation Engine
"""

try:
    import pytest
except ImportError:
    pytest = None
from src.nutrition.nutrition_engine import NutritionEstimationEngine

def test_nutrition_mass_calculation():
    engine = NutritionEstimationEngine()
    
    # 200 cm3 of raw apple: density ~0.81 g/cm3 -> mass ~162 g
    res = engine.estimate_portion("apple", volume_cm3=200.0, volume_std_cm3=10.0)
    
    assert res["canonical_id"] == "apple_fresh_raw"
    mean_mass = res["estimated_mass_grams"]["mean"]
    std_mass = res["estimated_mass_grams"]["std"]
    
    # Verify expected mass range
    assert 150.0 <= mean_mass <= 175.0
    # Verify variance is non-zero and reasonable
    assert 5.0 <= std_mass <= 20.0
    
    # Check 95% CI bounds
    ci = res["estimated_mass_grams"]["95_ci"]
    assert ci[0] < mean_mass < ci[1]
    
    # Check calories
    calories = res["estimated_portion_nutrients"]["energy_kcal"]["estimate"]
    assert 75.0 <= calories <= 95.0
    
    # Check that unmeasurable factors are flagged explicitly
    trans_fat = res["estimated_portion_nutrients"]["trans_fat_g"]
    assert trans_fat["status"] == "UNMEASURED_OR_HIDDEN_FACTOR"

def test_unknown_food_uncertainty():
    engine = NutritionEstimationEngine()
    res = engine.estimate_portion("unidentified_dish", volume_cm3=100.0, volume_std_cm3=10.0)
    
    assert res["canonical_id"] == "unknown_solid_food"
    # Mass std should be higher due to unknown prior
    assert res["estimated_mass_grams"]["std"] > 15.0
    # Protein should show missing data
    assert res["estimated_portion_nutrients"]["protein_g"]["estimate"] == "MISSING_DATA"
