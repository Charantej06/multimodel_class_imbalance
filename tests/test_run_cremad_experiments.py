"""
tests/test_run_cremad_experiments.py
====================================
Unit and validation tests for Step 9: Latents + RCS + Baselines across
balanced and long-tailed settings (CREMA-D).

Verifies:
1. File existence and non-emptiness of master_table.csv, all_runs_detail.csv,
   confusion_matrices.json, and all 4 rcs_grid_{setting}.csv files.
2. Exact row counts (72 rows in master_table, 216 in all_runs_detail, 216 confusion matrices).
3. Metric bounds: 0.0 <= F1, Bal-Acc, G-Mean, Accuracy <= 1.0; std >= 0.0.
4. Per-class F1 coverage for all 6 CREMA-D emotion classes.
5. Confusion matrix validity (6x6 matrix, sum of elements == test split size: 1065).
6. RCS PCA-clamping invariants across all imbalance ratios.
7. Verification that all cached latents are finite (no NaN / Inf).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results_cremad"
SETTINGS = ["balanced", "lt_rho10", "lt_rho20", "lt_rho50"]
REPRESENTATIONS = ["fusion_v2", "fusion_v2_nogate", "concat_MLP_raw_features"]
METHODS = [
    "no_rebalance",
    "class_weighted_CE",
    "random_oversample",
    "latent_SMOTE",
    "balanced_softmax",
    "RCS-Optimal",
    "RCS_balanced_softmax",
]
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
TEST_SET_SIZE = 1065  # 182 * 5 + 155


class TestCremadStep9Outputs:
    """Verifies all Step 9 experiment artifacts, metrics, and invariants."""

    def test_output_files_exist_and_non_empty(self):
        master_csv = RESULTS_DIR / "master_table.csv"
        runs_csv = RESULTS_DIR / "all_runs_detail.csv"
        cm_json = RESULTS_DIR / "confusion_matrices.json"

        assert master_csv.is_file(), f"Missing {master_csv}"
        assert master_csv.stat().st_size > 0, f"{master_csv} is empty"

        assert runs_csv.is_file(), f"Missing {runs_csv}"
        assert runs_csv.stat().st_size > 0, f"{runs_csv} is empty"

        assert cm_json.is_file(), f"Missing {cm_json}"
        assert cm_json.stat().st_size > 0, f"{cm_json} is empty"

        for s in SETTINGS:
            grid_csv = RESULTS_DIR / f"rcs_grid_{s}.csv"
            assert grid_csv.is_file(), f"Missing {grid_csv}"
            assert grid_csv.stat().st_size > 0, f"{grid_csv} is empty"

    def test_master_table_schema_and_counts(self):
        df = pd.read_csv(RESULTS_DIR / "master_table.csv")
        assert len(df) == 84, f"Expected exactly 84 rows (4 settings * 3 representations * 7 methods), got {len(df)}"

        assert set(df["setting"].unique()) == set(SETTINGS)
        assert set(df["representation"].unique()) == set(REPRESENTATIONS)
        assert set(df["method"].unique()) == set(METHODS)
        assert (df["num_seeds"] == 3).all(), "Every configuration must be evaluated across 3 seeds"

        # Check required columns
        metric_cols = ["val_macro_f1", "test_macro_f1", "balanced_accuracy", "gmean", "accuracy"]
        for m in metric_cols:
            assert f"{m}_mean" in df.columns
            assert f"{m}_std" in df.columns
            assert (df[f"{m}_mean"] >= 0.0).all() and (df[f"{m}_mean"] <= 1.0).all()
            assert (df[f"{m}_std"] >= 0.0).all()

        for emo in EMOTIONS:
            assert f"f1_{emo}_mean" in df.columns
            assert f"f1_{emo}_std" in df.columns
            assert (df[f"f1_{emo}_mean"] >= 0.0).all() and (df[f"f1_{emo}_mean"] <= 1.0).all()

        # No null or NaN
        assert not df.isnull().values.any(), "master_table.csv contains NaN values"

    def test_all_runs_detail_counts_and_metrics(self):
        df = pd.read_csv(RESULTS_DIR / "all_runs_detail.csv")
        assert len(df) == 252, f"Expected 252 individual runs (84 rows * 3 seeds), got {len(df)}"
        assert set(df["seed"].unique()) == {42, 43, 44}
        assert not df.isnull().values.any()

    def test_confusion_matrices_structure_and_sums(self):
        cm_path = RESULTS_DIR / "confusion_matrices.json"
        with open(cm_path, "r", encoding="utf-8") as f:
            cms = json.load(f)

        assert len(cms) == 252, f"Expected 252 confusion matrices, got {len(cms)}"

        for key, matrix in cms.items():
            mat = np.array(matrix)
            assert mat.shape == (6, 6), f"Confusion matrix for {key} has shape {mat.shape}, expected (6, 6)"
            assert np.all(mat >= 0), f"Negative values in confusion matrix for {key}"
            assert np.sum(mat) == TEST_SET_SIZE, (
                f"Sum of confusion matrix elements for {key} is {np.sum(mat)}, expected {TEST_SET_SIZE}"
            )

    def test_rcs_grid_pca_clamping_invariants(self):
        # Setting minimum class counts:
        # balanced: min_class = 764 -> max_pca = 764 // 4 = 191
        # lt_rho10: min_class = 89  -> max_pca = 89 // 4 = 22
        # lt_rho20: min_class = 45  -> max_pca = 45 // 4 = 11
        # lt_rho50: min_class = 18  -> max_pca = 18 // 4 = 4
        expected_max_pca = {
            "balanced": 191,
            "lt_rho10": 22,
            "lt_rho20": 11,
            "lt_rho50": 4,
        }

        for s in SETTINGS:
            df = pd.read_csv(RESULTS_DIR / f"rcs_grid_{s}.csv")
            assert len(df) > 0
            max_pca = expected_max_pca[s]
            assert (df["clamped_pca_dim"] <= max_pca).all(), (
                f"Clamped pca_dim in {s} exceeds maximum allowed {max_pca}"
            )
            assert (df["val_macro_f1"] >= 0.0).all() and (df["val_macro_f1"] <= 1.0).all()
            assert (df["val_accuracy"] >= 0.0).all() and (df["val_accuracy"] <= 1.0).all()

    def test_cached_latents_validity(self):
        latents_dir = RESULTS_DIR / "latents"
        assert latents_dir.is_dir()

        latent_files = list(latents_dir.glob("*.npz"))
        assert len(latent_files) >= 24, f"Expected at least 24 cached latent files, found {len(latent_files)}"

        for lf in latent_files:
            data = np.load(lf)
            for split in ["Z_tr", "Z_va", "Z_te"]:
                assert split in data, f"Missing {split} in {lf.name}"
                arr = data[split]
                assert not np.isnan(arr).any(), f"NaN in {lf.name} [{split}]"
                assert not np.isinf(arr).any(), f"Inf in {lf.name} [{split}]"
                assert arr.shape[1] == 256, f"Latent dim is {arr.shape[1]}, expected 256 in {lf.name}"

