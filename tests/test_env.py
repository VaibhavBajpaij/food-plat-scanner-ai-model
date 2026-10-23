"""
Tests for Environment and Hardware Verification
"""

import unittest
from src.utils.verify_env import run_environment_audit

class TestEnvironment(unittest.TestCase):
    def test_environment_structure(self):
        audit = run_environment_audit()
        self.assertIn("status", audit)
        self.assertTrue("disk_info" in audit or "disk_info_d_drive" in audit)
        disk_info = audit.get("disk_info") or audit.get("disk_info_d_drive", {})
        free_gb = disk_info.get("free_gb", 0)
        self.assertGreaterEqual(free_gb, 4.0, f"Insufficient disk space on active environment: {free_gb} GB")

if __name__ == "__main__":
    unittest.main()
