"""
Synthetic 3D Geometry and Volume Calculator
Provides ground-truth geometric shapes and volumetric estimation algorithms.
"""

import math
from typing import Dict, Any

class SyntheticVolumeEstimator:
    @staticmethod
    def calculate_sphere_volume(radius_cm: float) -> float:
        """
        Exact volume: (4/3) * pi * r^3
        """
        return (4.0 / 3.0) * math.pi * (radius_cm ** 3)

    @staticmethod
    def calculate_cylinder_volume(radius_cm: float, height_cm: float) -> float:
        """
        Exact volume: pi * r^2 * h
        """
        return math.pi * (radius_cm ** 2) * height_cm

    @staticmethod
    def calculate_ellipsoid_volume(a_cm: float, b_cm: float, c_cm: float) -> float:
        """
        Exact volume: (4/3) * pi * a * b * c
        Useful for modeling oblong fruits (bananas, mangoes, bread).
        """
        return (4.0 / 3.0) * math.pi * a * b * c

    @staticmethod
    def estimate_plate_scale(plate_diameter_px: float, nominal_plate_diameter_cm: float = 24.0) -> float:
        """
        Returns metric scale factor in pixels per centimeter (px/cm).
        """
        if plate_diameter_px <= 0 or nominal_plate_diameter_cm <= 0:
            raise ValueError("Diameters must be positive numbers.")
        return plate_diameter_px / nominal_plate_diameter_cm
