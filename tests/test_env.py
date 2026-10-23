"""
Tests for Environment and Hardware Verification
"""

import unittest
from src.utils.verify_env import run_environment_audit

class TestEnvironment(unittest.TestCase):
    def test_environment_structure(self):
        audit = run_environment_audit()
        self.assertIn("status", audit)
        self.assertIn("disk_info_d_drive", audit)
        free_gb = audit["disk_info_d_drive"].get("free_gb", 0)
        self.assertGreaterEqual(free_gb, 15.0, f"Insufficient disk space on D: drive: {free_gb} GB")

if __name__ == "__main__":
    unittest.main()
