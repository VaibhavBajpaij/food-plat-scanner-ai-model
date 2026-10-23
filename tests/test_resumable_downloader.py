"""
Unit Tests for Resumable Downloader, 9 Candidate Adapters & Compliance Guardrails
"""

import os
import sys
import unittest
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.resumable_downloader import (
    ComplianceViolationError,
    AccessVerificationError,
    EnhancedIndianFoodAdapter,
    IndianFoodsDatasetAdapter,
    BDFood33Adapter,
    FoodBDAdapter,
    SpiceVisionAdapter,
    IndianSpicesAdapter,
    FoodLensClosedAdapter,
    KhanaQuarantinedAdapter,
    RasoiAdapter,
    CloudDatasetOrchestrator
)

def test_compliance_guardrail_blocks_noncommercial():
    # Khana is CC BY-NC-ND 4.0
    khana = KhanaQuarantinedAdapter()
    assert not khana.is_commercial_approved, "Khana must not be marked commercial approved!"
    try:
        khana.acquire()
        assert False, "Acquiring Khana must raise ComplianceViolationError!"
    except ComplianceViolationError:
        pass  # Expected behavior

    # FoodLens is Closed / Proprietary
    foodlens = FoodLensClosedAdapter()
    assert not foodlens.is_commercial_approved, "FoodLens must not be marked commercial approved!"
    try:
        foodlens.acquire()
        assert False, "Acquiring FoodLens must raise AccessVerificationError!"
    except AccessVerificationError:
        pass  # Expected behavior

    # SpiceVision is unverified access
    spicevision = SpiceVisionAdapter()
    assert not spicevision.is_commercial_approved, "SpiceVision must not be marked commercial approved!"
    try:
        spicevision.acquire()
        assert False, "Acquiring SpiceVision must raise AccessVerificationError!"
    except AccessVerificationError:
        pass  # Expected behavior

    # Indian Foods requires review if curated flag not passed
    ind_foods = IndianFoodsDatasetAdapter(allow_reviewed_curated=False)
    assert not ind_foods.is_commercial_approved, "Indian Foods must require review by default!"
    try:
        ind_foods.acquire()
        assert False, "Acquiring unreviewed Indian Foods must raise ComplianceViolationError!"
    except ComplianceViolationError:
        pass  # Expected behavior

def test_approved_adapters_planning():
    sohl = EnhancedIndianFoodAdapter()
    plan_sohl = sohl.plan()
    assert plan_sohl["commercial_approved"], "SohlHealth must be commercial approved!"
    assert plan_sohl["total_images"] == 15404, "SohlHealth image count must be 15,404"
    assert len(plan_sohl["shards"]) == 5, "SohlHealth must have 5 parquet shards"

    bd = BDFood33Adapter()
    plan_bd = bd.plan()
    assert plan_bd["commercial_approved"], "BD-Food33 must be commercial approved!"
    assert plan_bd["licence"] == "CC0-1.0-Universal", "BD-Food33 must have CC0 licence"

    foodbd = FoodBDAdapter()
    plan_foodbd = foodbd.plan()
    assert plan_foodbd["commercial_approved"], "FoodBD must be commercial approved!"

    spices = IndianSpicesAdapter()
    plan_spices = spices.plan()
    assert plan_spices["commercial_approved"], "Indian Spices must be commercial approved!"
    assert plan_spices["total_images"] == 10991, "Indian Spices must have 10,991 images"

    rasoi = RasoiAdapter()
    plan_rasoi = rasoi.plan()
    assert plan_rasoi["commercial_approved"], "RASOI must be commercial approved!"

def test_orchestrator_covers_all_nine_datasets():
    orch = CloudDatasetOrchestrator()
    assert len(orch.adapters) == 9, f"Orchestrator must cover all 9 candidate datasets, found {len(orch.adapters)}"
    plan = orch.generate_acquisition_plan()

    assert plan["total_approved_images"] >= 10000, f"Total approved images must exceed 10,000, got {plan['total_approved_images']}"
    assert len(plan["approved_commercial_sources"]) == 5, f"Expected 5 approved sources, got {len(plan['approved_commercial_sources'])}"
    assert len(plan["requires_review_sources"]) == 1, f"Expected 1 requires-review source, got {len(plan['requires_review_sources'])}"
    assert len(plan["excluded_non_commercial_sources"]) == 1, f"Expected 1 excluded non-commercial source, got {len(plan['excluded_non_commercial_sources'])}"
    assert len(plan["access_unverified_sources"]) == 2, f"Expected 2 access-unverified sources, got {len(plan['access_unverified_sources'])}"

if __name__ == "__main__":
    test_compliance_guardrail_blocks_noncommercial()
    test_approved_adapters_planning()
    test_orchestrator_covers_all_nine_datasets()
    print("All Resumable Downloader tests covering all 9 datasets passed!")
