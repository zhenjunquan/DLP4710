"""
Tests for DLP4710 Pattern Binary Building, Inspection, Validation, and Burning.
"""

import os
import tempfile
import unittest

from dlp_common.hardware.dlp_controller import DLP4710Controller
from dlp_common.hardware.patn_toolkit import (
    build_gray_and_phase_shift_bin,
    build_standard_phase_shift_bin,
    inspect_patn,
    validate_patn,
)


class TestPatternBinToolkit(unittest.TestCase):

    def test_build_standard_phase_shift_bin(self):
        bin_bytes, info = build_standard_phase_shift_bin(
            frequencies=(70, 64, 59),
            steps=4,
            exposure_us=20000,
            dark_us=5000,
            led="blue",
        )
        self.assertGreater(len(bin_bytes), 1000)
        self.assertTrue(bin_bytes.startswith(b"PATN"))

        # In-memory validation
        ok, issues, report = validate_patn(bin_bytes)
        self.assertTrue(ok, f"Validation failed with issues: {issues}")
        self.assertEqual(report["total_patterns"], 12)
        self.assertEqual(report["order_count"], 2)  # 2 sets (8 + 4 = 12 patterns)

    def test_build_gray_and_phase_shift_bin(self):
        bin_bytes, info = build_gray_and_phase_shift_bin(
            gray_bits=4,
            frequencies=(64,),
            steps=4,
            exposure_us=15000,
            dark_us=3000,
            led="green",
        )
        self.assertGreater(len(bin_bytes), 1000)
        self.assertTrue(bin_bytes.startswith(b"PATN"))

        ok, issues, report = validate_patn(bin_bytes)
        self.assertTrue(ok, f"Validation failed with issues: {issues}")
        self.assertGreaterEqual(report["total_patterns"], 12)

    def test_inspect_patn(self):
        bin_bytes, _ = build_standard_phase_shift_bin(frequencies=(64,), steps=4)
        info = inspect_patn(bin_bytes)
        self.assertEqual(info["format"], "PATN")
        self.assertIn("sha256", info)
        self.assertGreater(len(info["pattern_sets"]), 0)
        self.assertGreater(len(info["order"]), 0)

    def test_validate_corrupt_data(self):
        corrupt = b"INVALID_HEADER_DATA_12345"
        ok, issues, report = validate_patn(corrupt)
        self.assertFalse(ok)
        self.assertGreater(len(issues), 0)

    def test_burn_and_verify_bin_mock(self):
        bin_bytes, _ = build_standard_phase_shift_bin()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(bin_bytes)
            tmp_path = f.name

        try:
            dlp = DLP4710Controller(dll_path="nonexistent.dll")
            self.assertTrue(dlp.is_mock)
            dlp.connect()

            stages = []

            def progress_cb(pct, msg):
                stages.append((pct, msg))

            ok, report, msg = dlp.burn_and_verify_bin(
                bin_path=tmp_path,
                verify_projection=True,
                progress_callback=progress_cb,
            )
            self.assertTrue(ok, f"burn_and_verify_bin failed: {msg}")
            self.assertTrue(report["pre_burn_passed"])
            self.assertTrue(report["flash_write_passed"])
            self.assertTrue(report["order_table_passed"])
            self.assertTrue(report["readback_passed"])
            self.assertTrue(report["dry_run_passed"])
            self.assertGreaterEqual(len(stages), 5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
