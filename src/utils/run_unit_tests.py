"""
Standalone Test Runner for Verification
Executes taxonomy, nutrition uncertainty, synthetic geometry, environment, and dataset V2 integrity tests.
"""

import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def run_all_tests():
    from tests.test_taxonomy import test_canonical_resolution, test_density_retrieval
    from tests.test_nutrition import test_nutrition_mass_calculation, test_unknown_food_uncertainty
    from tests.test_reconstruction_synthetic import test_sphere_volume, test_plate_scale_estimation
    from tests.test_env import TestEnvironment
    from tests.test_dataset_v2 import test_model_head_adaptation, test_v2_manifest_and_leakage
    from tests.test_dataset_v3 import test_model_candidates_forward_pass, test_v3_manifest_and_leakage
    from tests.test_kaggle_pipeline import (
        test_dhash_and_hamming_distance,
        test_image_validation_integrity,
        test_license_classification_logic,
        test_mobilenet_v3_10class_shape
    )
    from tests.test_resumable_downloader import (
        test_compliance_guardrail_blocks_noncommercial,
        test_approved_adapters_planning,
        test_orchestrator_covers_all_nine_datasets
    )
    from tests.test_parquet_food_loader import (
        test_parquet_image_decoding,
        test_get_split_from_path,
        test_43class_vocabulary,
        test_split_and_leakage_audit,
        test_legacy_dataset_guardrail_rejection,
        test_parquet_synthetic_roundtrip_and_smoke_test,
        test_mobilenet_v3_43class_shape
    )
    from tests.test_repaired_splits_and_evaluation import (
        test_manifest_integrity,
        test_label_mapping_and_checkpoint_compatibility,
        test_class_coverage_across_all_splits,
        test_duplicate_leakage_and_split_disjointness,
        test_perceptual_cross_split_leakage,
        test_metrics_calculation_and_wilson_ci,
        test_checkpoint_contamination_detection
    )
    from tests.test_clean_retraining_and_readiness import (
        test_clean_imagenet_initialization_not_contaminated_checkpoint,
        test_43class_classifier_head_structure,
        test_repaired_manifest_duplicate_group_isolation,
        test_test_split_isolation_guardrail,
        test_path_validation_safety_stop,
        test_deployment_readiness_metrics_and_guardrail
    )

    env_tester = TestEnvironment()

    tests = [
        ("test_canonical_resolution", test_canonical_resolution),
        ("test_density_retrieval", test_density_retrieval),
        ("test_nutrition_mass_calculation", test_nutrition_mass_calculation),
        ("test_unknown_food_uncertainty", test_unknown_food_uncertainty),
        ("test_sphere_volume", test_sphere_volume),
        ("test_plate_scale_estimation", test_plate_scale_estimation),
        ("test_environment_structure", env_tester.test_environment_structure),
        ("test_model_head_adaptation", test_model_head_adaptation),
        ("test_v2_manifest_and_leakage", test_v2_manifest_and_leakage),
        ("test_model_candidates_forward_pass", test_model_candidates_forward_pass),
        ("test_v3_manifest_and_leakage", test_v3_manifest_and_leakage),
        ("test_dhash_and_hamming_distance", test_dhash_and_hamming_distance),
        ("test_image_validation_integrity", test_image_validation_integrity),
        ("test_license_classification_logic", test_license_classification_logic),
        ("test_mobilenet_v3_10class_shape", test_mobilenet_v3_10class_shape),
        ("test_compliance_guardrail_blocks_noncommercial", test_compliance_guardrail_blocks_noncommercial),
        ("test_approved_adapters_planning", test_approved_adapters_planning),
        ("test_orchestrator_covers_all_nine_datasets", test_orchestrator_covers_all_nine_datasets),
        ("test_parquet_image_decoding", test_parquet_image_decoding),
        ("test_get_split_from_path", test_get_split_from_path),
        ("test_43class_vocabulary", test_43class_vocabulary),
        ("test_split_and_leakage_audit", test_split_and_leakage_audit),
        ("test_legacy_dataset_guardrail_rejection", test_legacy_dataset_guardrail_rejection),
        ("test_parquet_synthetic_roundtrip_and_smoke_test", test_parquet_synthetic_roundtrip_and_smoke_test),
        ("test_mobilenet_v3_43class_shape", test_mobilenet_v3_43class_shape),
        ("test_manifest_integrity", test_manifest_integrity),
        ("test_label_mapping_and_checkpoint_compatibility", test_label_mapping_and_checkpoint_compatibility),
        ("test_class_coverage_across_all_splits", test_class_coverage_across_all_splits),
        ("test_duplicate_leakage_and_split_disjointness", test_duplicate_leakage_and_split_disjointness),
        ("test_perceptual_cross_split_leakage", test_perceptual_cross_split_leakage),
        ("test_metrics_calculation_and_wilson_ci", test_metrics_calculation_and_wilson_ci),
        ("test_checkpoint_contamination_detection", test_checkpoint_contamination_detection),
        ("test_clean_imagenet_initialization_not_contaminated_checkpoint", test_clean_imagenet_initialization_not_contaminated_checkpoint),
        ("test_43class_classifier_head_structure", test_43class_classifier_head_structure),
        ("test_repaired_manifest_duplicate_group_isolation", test_repaired_manifest_duplicate_group_isolation),
        ("test_test_split_isolation_guardrail", test_test_split_isolation_guardrail),
        ("test_path_validation_safety_stop", test_path_validation_safety_stop),
        ("test_deployment_readiness_metrics_and_guardrail", test_deployment_readiness_metrics_and_guardrail),
    ]

    passed = 0
    failed = 0
    print("=" * 65)
    print("RUNNING ALL UNIT & INTEGRITY TESTS")
    print("=" * 65)

    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {str(e)}")
            traceback.print_exc()
            failed += 1

    print("=" * 65)
    print(f"Results: {passed} passed, {failed} failed.")
    print("=" * 65)
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
