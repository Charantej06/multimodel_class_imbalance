"""
tests/test_food101_lt.py
========================
Test suite for the Long-Tailed Food-101 profile generator and split builder.
"""
import sys
from pathlib import Path
import json
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None

from food_101_imbalanced_making.food101_profile import make_counts, check_profile, PRESETS
from food_101_imbalanced_making.build_split import build_food101_lt_split, create_mock_food101_meta
from food_101_imbalanced_making.verify import verify_food101_lt


def test_make_counts_main():
    counts = make_counts(n_max=700, n_min=40, C=101)
    assert len(counts) == 101
    assert counts[0] == 700
    assert counts[-1] == 40
    # Check monotonicity
    assert all(counts[i] >= counts[i + 1] for i in range(100))


def test_check_profile_main():
    counts = make_counts(n_max=700, n_min=40, C=101)
    info = check_profile(counts, eta=7, k=5, allow_below_paper_min=False, verbose=False)
    assert info["zeta"] == 100
    assert info["n_gmm_classes"] == 68
    assert info["n_minority_classes"] == 33
    assert info["predicted_gmm_components"] == 248
    assert info["imbalance_ratio"] == 17.5


def test_stress_preset_assertion():
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


def test_build_split_mock(tmp_path):
    mock_root = tmp_path / "mock_food101"
    create_mock_food101_meta(mock_root)

    out_dir = tmp_path / "output_food101_lt"
    res = build_food101_lt_split(
        food101_root=mock_root,
        output_dir=out_dir,
        seed=42,
        n_max=700,
        n_min=40,
        C=101,
        eta=7,
        k=5,
        allow_below_paper_min=False,
        preset_name="main",
    )

    assert res["manifest_csv"].exists()
    assert res["class_map_json"].exists()
    assert res["config_json"].exists()

    with open(res["config_json"]) as f:
        cfg = json.load(f)
    assert cfg["n_train_total"] == 23430
    assert cfg["n_val_total"] == 5050
    assert cfg["n_test_total"] == 25250
    assert cfg["n_total"] == 53730

    with open(res["class_map_json"]) as f:
        cmap = json.load(f)
    assert cmap["0"]["train_count"] == 700
    assert cmap["100"]["train_count"] == 40


def test_verify_food101_lt_success():
    data_dir = Path("data/food101_lt")
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
    data_dir = Path("data/food101_lt")
    # Intentionally trigger shape assert failure by requesting n_max=800 against the 700-profile tensors
    failed = False
    try:
        verify_food101_lt(
            data_dir=data_dir,
            report_dir=data_dir / "report",
            n_max=800,
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
    print("ALL TESTS PASSED SUCCESSFULLY!")
