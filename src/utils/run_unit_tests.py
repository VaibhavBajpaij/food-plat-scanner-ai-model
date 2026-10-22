"""
Standalone Test Runner for Phase 0 Verification
Executes taxonomy, nutrition uncertainty, and synthetic geometry tests without third-party test runners.
"""

import sys
import traceback

def run_all_tests():
    from tests.test_taxonomy import test_canonical_resolution, test_density_retrieval
    from tests.test_nutrition import test_nutrition_mass_calculation, test_unknown_food_uncertainty
    from tests.test_reconstruction_synthetic import test_sphere_volume, test_plate_scale_estimation

    tests = [
        ("test_canonical_resolution", test_canonical_resolution),
        ("test_density_retrieval", test_density_retrieval),
        ("test_nutrition_mass_calculation", test_nutrition_mass_calculation),
        ("test_unknown_food_uncertainty", test_unknown_food_uncertainty),
        ("test_sphere_volume", test_sphere_volume),
        ("test_plate_scale_estimation", test_plate_scale_estimation),
    ]

    passed = 0
    failed = 0
    print("=" * 60)
    print("RUNNING PHASE 0 UNIT TESTS")
    print("=" * 60)

    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {str(e)}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed.")
    print("=" * 60)
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
