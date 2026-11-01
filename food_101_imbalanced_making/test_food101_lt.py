"""
food_101_imbalanced_making/test_food101_lt.py
=============================================
Automated test and verification suite for the Long-Tailed Food-101
pipeline: profile generation, dataset splitting, tensor reloading, and verification assertions.
"""

import sys
import os
import json
import csv
from pathlib import Path

# Support running directly within food_101_imbalanced_making or from repository root
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    from food101_profile import make_counts, check_profile, PRESETS
    from build_split import build_food101_lt_split, create_mock_food101_meta
    from verify import verify_food101_lt
except ImportError:
    from food_101_imbalanced_making.food101_profile import make_counts, check_profile, PRESETS
    from food_101_imbalanced_making.build_split import build_food101_lt_split, create_mock_food101_meta
    from food_101_imbalanced_making.verify import verify_food101_lt


def test_make_counts_main():
    """Validates exponential decay formula across 101 classes."""
    counts = make_counts(n_max=700, n_min=40, C=101)
    assert len(counts) == 101, f"Expected 101 classes, got {len(counts)}"
    assert counts[0] == 700, f"Class 0 count must be 700, got {counts[0]}"
    assert counts[-1] == 40, f"Class 100 count must be 40, got {counts[-1]}"
    assert sum(counts) == 23430, f"Expected total 23,430, got {sum(counts)}"
    # Monotonicity check
    assert all(counts[i] >= counts[i + 1] for i in range(100)), "Counts are not strictly non-increasing"


def test_check_profile_main():
    """Validates all 6 mathematical assumptions of the RCS GMM pipeline."""
    counts = make_counts(n_max=700, n_min=40, C=101)
    info = check_profile(counts, eta=7, k=5, allow_below_paper_min=False, verbose=False)
    assert info["zeta"] == 100
    assert info["n_gmm_classes"] == 68
    assert info["n_minority_classes"] == 33
    assert info["predicted_gmm_components"] == 248
    assert info["imbalance_ratio"] == 17.5


def test_stress_preset_assertion():
    """Validates that n_min < 40 raises AssertionError unless explicitly overridden."""
    counts = make_counts(n_max=700, n_min=20, C=101)
    failed = False
    try:
        check_profile(counts, eta=7, k=5, allow_below_paper_min=False, verbose=False)
    except AssertionError as e:
        failed = True
        assert "Assert failed (2)" in str(e)
    assert failed, "Expected check_profile to raise AssertionError without allow_below_paper_min=True"

    # With override, it should pass
    info = check_profile(counts, eta=7, k=5, allow_below_paper_min=True, verbose=False)
    assert info["n_min"] == 20
    assert info["imbalance_ratio"] == 35.0


def test_verify_food101_lt_success():
    """Validates full end-to-end verification suite on generated dataset."""
    data_dir = REPO_ROOT / "data" / "food101_lt"
    report_dir = data_dir / "report"
    summary = verify_food101_lt(
        data_dir=data_dir,
        report_dir=report_dir,
        mock=False,
    )
    assert summary["imbalance_ratio"] == 17.5
    assert summary["total_train"] == 23430
    assert summary["total_val"] == 5050
    assert summary["total_test"] == 25250
    assert Path(summary["bar_chart_path"]).exists()
    assert Path(summary["largest_grid_path"]).exists()
    assert Path(summary["smallest_grid_path"]).exists()


def test_verify_assertion_failure():
    """Validates that verify_food101_lt strictly raises AssertionError and fails non-zero on corrupt parameters."""
    data_dir = REPO_ROOT / "data" / "food101_lt"
    failed = False
    try:
        verify_food101_lt(
            data_dir=data_dir,
            report_dir=data_dir / "report",
            n_max=800,  # Deliberate parameter mismatch against 700-profile tensors
            mock=False,
        )
    except AssertionError as e:
        failed = True
        assert "Assert failed" in str(e)
    assert failed, "Expected verify_food101_lt to raise AssertionError on mismatched parameters"


if __name__ == "__main__":
    test_make_counts_main()
    print("test_make_counts_main: PASS")
    test_check_profile_main()
    print("test_check_profile_main: PASS")
    test_stress_preset_assertion()
    print("test_stress_preset_assertion: PASS")
    test_verify_food101_lt_success()
    print("test_verify_food101_lt_success: PASS")
    test_verify_assertion_failure()
    print("test_verify_assertion_failure: PASS")
    print("ALL TESTS IN food_101_imbalanced_making PASSED SUCCESSFULLY!")

