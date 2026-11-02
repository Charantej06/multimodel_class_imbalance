"""
tests/test_feature_sets_cremad.py
=================================
Unit and regression tests for CREMA-D feature sets and diagnostic probing (Step 7b).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestFeatureProbeOutputs:
    def test_probe_csv_exists_and_valid(self):
        csv_path = REPO_ROOT / "cremad/feature_probe.csv"
        assert csv_path.is_file(), f"Missing probe CSV: {csv_path}"
        df = pd.read_csv(csv_path)

        expected_cols = [
            "feature_set", "dim", "val_macro_f1", "val_accuracy",
            "test_macro_f1", "test_accuracy", "f1_anger", "f1_disgust",
            "f1_fear", "f1_happy", "f1_neutral", "f1_sad", "chosen_video_variant"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        assert len(df) == 4
        assert "audio_only" in df["feature_set"].values
        assert "video_only_mean" in df["feature_set"].values
        assert "video_only_mean_std" in df["feature_set"].values

        # video_only_mean_std must outperform video_only_mean on validation Macro-F1
        vmean_row = df[df["feature_set"] == "video_only_mean"].iloc[0]
        vstd_row = df[df["feature_set"] == "video_only_mean_std"].iloc[0]
        assert vstd_row["val_macro_f1"] > vmean_row["val_macro_f1"]

        # Concat must outperform audio-only on validation Macro-F1
        aud_row = df[df["feature_set"] == "audio_only"].iloc[0]
        cat_row = df[df["feature_set"].str.startswith("concat_")].iloc[0]
        assert cat_row["val_macro_f1"] > aud_row["val_macro_f1"]


@pytest.fixture(scope="module")
def balanced_data():
    base_dir = REPO_ROOT / "data/features_cremad"
    splits = {}
    for s in ["train", "val", "test"]:
        p = base_dir / f"{s}_features.npz"
        assert p.is_file(), f"Missing feature file: {p}"
        splits[s] = np.load(p)
    return splits


class TestBalancedFeatureSet:

    def test_keys_and_shapes(self, balanced_data):
        expected_counts = {"train": 5229, "val": 1148, "test": 1065}
        for s, d in balanced_data.items():
            N = expected_counts[s]
            assert d["X_audio"].shape == (N, 768)
            assert d["X_video"].shape == (N, 1536)
            assert d["y"].shape == (N,)
            assert d["ids"].shape == (N,)
            assert d["actor"].shape == (N,)

            assert d["X_audio"].dtype == np.float32
            assert d["X_video"].dtype == np.float32

    def test_no_nan_or_inf(self, balanced_data):
        for s, d in balanced_data.items():
            assert not np.isnan(d["X_audio"]).any()
            assert not np.isinf(d["X_audio"]).any()
            assert not np.isnan(d["X_video"]).any()
            assert not np.isinf(d["X_video"]).any()

    def test_actor_isolation(self, balanced_data):
        tr_actors = set(balanced_data["train"]["actor"])
        va_actors = set(balanced_data["val"]["actor"])
        te_actors = set(balanced_data["test"]["actor"])

        assert len(tr_actors & va_actors) == 0
        assert len(tr_actors & te_actors) == 0
        assert len(va_actors & te_actors) == 0

    def test_labels_in_bounds(self, balanced_data):
        for s, d in balanced_data.items():
            assert d["y"].min() >= 0
            assert d["y"].max() <= 5
            # All 6 classes must be present
            assert len(np.unique(d["y"])) == 6


class TestLongTailedFeatureSets:
    @pytest.mark.parametrize("rho, expected_train_n", [
        (10, 2267),
        (20, 1927),
        (50, 1630),
    ])
    def test_lt_split_properties(self, rho, expected_train_n):
        lt_dir = REPO_ROOT / f"data/features_cremad_lt_rho{rho}"
        assert lt_dir.is_dir()

        tr = np.load(lt_dir / "train_features.npz")
        va = np.load(lt_dir / "val_features.npz")
        te = np.load(lt_dir / "test_features.npz")

        assert len(tr["y"]) == expected_train_n
        assert len(va["y"]) == 1148
        assert len(te["y"]) == 1065

        assert tr["X_audio"].shape == (expected_train_n, 768)
        assert tr["X_video"].shape == (expected_train_n, 1536)

        # No NaNs
        assert not np.isnan(tr["X_audio"]).any()
        assert not np.isnan(tr["X_video"]).any()

        # Actor isolation
        tr_act = set(tr["actor"])
        va_act = set(va["actor"])
        te_act = set(te["actor"])
        assert len(tr_act & va_act) == 0
        assert len(tr_act & te_act) == 0
