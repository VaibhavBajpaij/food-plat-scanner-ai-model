"""
Tests for Synthetic 3D Geometry and Volume Calculation
"""

import math
try:
    import pytest
except ImportError:
    pytest = None
from src.reconstruction.synthetic_geometry import SyntheticVolumeEstimator

def approx_equal(a, b, rel=1e-2):
    return abs(a - b) <= rel * max(abs(a), abs(b))

def test_sphere_volume():
    # Apple with radius 3.8 cm (diameter 7.6 cm)
    # Expected: (4/3) * pi * 3.8^3 = ~229.85 cm3
    vol = SyntheticVolumeEstimator.calculate_sphere_volume(3.8)
    expected = (4.0 / 3.0) * math.pi * (3.8 ** 3)
    assert approx_equal(vol, expected)
    assert 225.0 <= vol <= 235.0

def test_plate_scale_estimation():
    # 380 px across a 24.0 cm plate
    scale = SyntheticVolumeEstimator.estimate_plate_scale(380.0, 24.0)
    assert approx_equal(scale, 15.833)

    # Recover diameter in cm from pixel measurement: 120 px / 15.833 px/cm = ~7.58 cm
    food_diameter_cm = 120.0 / scale
    assert 7.5 <= food_diameter_cm <= 7.7
