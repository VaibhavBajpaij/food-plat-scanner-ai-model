"""
Tests for Environment and Hardware Verification
"""

import pytest
from src.utils.verify_env import run_environment_audit

def test_environment_structure():
    audit = run_environment_audit()
    assert "status" in audit
    assert "disk_info_d_drive" in audit
    # Ensure D: drive has at least 15 GB free
    free_gb = audit["disk_info_d_drive"].get("free_gb", 0)
    assert free_gb >= 15.0, f"Insufficient disk space on D: drive: {free_gb} GB"
