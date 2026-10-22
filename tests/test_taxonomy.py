"""
Tests for Food Taxonomy Engine
"""

try:
    import pytest
except ImportError:
    pytest = None
from src.data.taxonomy import FoodTaxonomyEngine

def test_canonical_resolution():
    engine = FoodTaxonomyEngine()
    
    # Direct match
    cid, info = engine.resolve_food("apple_fresh_raw")
    assert cid == "apple_fresh_raw"
    assert "Apple" in info["canonical_name_en"]

    # Multilingual alias match (Hindi)
    cid_hi, _ = engine.resolve_food("seb")
    assert cid_hi == "apple_fresh_raw"

    # Multilingual alias match (Chinese)
    cid_zh, _ = engine.resolve_food("mifan")
    assert cid_zh == "rice_white_cooked"

    # Unknown fallback
    cid_unk, info_unk = engine.resolve_food("exotic_unseen_plant_berry_xyz")
    assert cid_unk == "unknown_solid_food"

def test_density_retrieval():
    engine = FoodTaxonomyEngine()
    
    mean_d, std_d = engine.get_density("apple_fresh_raw")
    assert 0.70 <= mean_d <= 0.90
    assert 0.01 <= std_d <= 0.10

    # Test chicken breast density
    mean_c, std_c = engine.get_density("chicken_breast_grilled")
    assert 1.00 <= mean_c <= 1.15
