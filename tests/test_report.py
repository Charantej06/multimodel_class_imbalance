"""
tests/test_report.py
====================
Unit tests for Step 10: Aggregated report, statistical tests, gate weight analysis,
leakage verification, and pipeline execution script.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results_cremad"
SETTINGS = ["balanced", "lt_rho10", "lt_rho20", "lt_rho50"]
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


class TestReportOutputs:
    """Verifies all Step 10 report artifacts, visualizations, and summary files."""

    def test_report_md_exists_and_content_complete(self):
        report_path = RESULTS_DIR / "REPORT.md"
        assert report_path.is_file(), f"Missing {report_path}"
        assert report_path.stat().st_size > 1000, "REPORT.md is too short or empty"

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check required sections
        assert "Executive Summary & Honest Method Rankings" in content
        assert "Complete Master Benchmark Table" in content
        assert "Statistical Significance Testing" in content
        assert "Multimodal Gate Weights Analysis per Emotion" in content
        assert "Comprehensive Ablation Studies" in content
        assert "In-Depth Scientific Discussion" in content

        # Check that RCS is honestly ranked
        assert "Rank 5 of 6" in content
        assert "Balanced Softmax" in content
        assert "Double-compensation" in content

    def test_visualizations_exist_and_non_empty(self):
        for img_name in [
            "gate_weights.png",
            "confusion_matrices_all_settings.png",
            "cm_comparison_rho50.png",
        ]:
            img_path = RESULTS_DIR / img_name
            assert img_path.is_file(), f"Missing visualization: {img_path}"
            assert img_path.stat().st_size > 5000, f"Image {img_path} is suspiciously small"

    def test_gate_weights_json_validity(self):
        gw_path = RESULTS_DIR / "gate_weights.json"
        assert gw_path.is_file(), f"Missing {gw_path}"
        with open(gw_path, "r", encoding="utf-8") as f:
            gw = json.load(f)

        for s in SETTINGS:
            assert s in gw, f"Setting {s} missing from gate_weights.json"
            for emo in EMOTIONS:
                assert emo in gw[s], f"Emotion {emo} missing from {s} in gate_weights.json"
                a_m = gw[s][emo]["audio_mean"]
                v_m = gw[s][emo]["video_mean"]
                assert 0.0 <= a_m <= 1.0, f"Invalid audio gate mean: {a_m}"
                assert 0.0 <= v_m <= 1.0, f"Invalid video gate mean: {v_m}"
                assert abs((a_m + v_m) - 1.0) < 1e-4, f"Gate means do not sum to 1.0: {a_m + v_m}"

    def test_test_predictions_json_count(self):
        tp_path = RESULTS_DIR / "test_predictions.json"
        assert tp_path.is_file(), f"Missing {tp_path}"
        with open(tp_path, "r", encoding="utf-8") as f:
            tp = json.load(f)

        # 4 settings * 3 representations * 7 methods * 3 seeds = 252
        assert len(tp) == 252, f"Expected 252 prediction runs, got {len(tp)}"
        for k, preds in tp.items():
            assert len(preds) == 1065, f"Prediction run {k} length is {len(preds)}, expected 1065"

    def test_run_all_cremad_script_exists(self):
        script_path = REPO_ROOT / "run_all_cremad.sh"
        assert script_path.is_file(), f"Missing {script_path}"
        assert script_path.stat().st_size > 200
        with open(script_path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "scripts/verify_cremad_data.py" in text
        assert "scripts/run_cremad_experiments.py" in text
        assert "scripts/verify_leakage.py" in text
        assert "scripts/aggregate_cremad.py" in text

