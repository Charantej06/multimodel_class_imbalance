"""
tests/test_prepare_cremad.py
============================
Unit and integration tests for CREMA-D manifest generation and actor-level splitting (Step 3).

Verifies:
1. Manifest schema and format (id, actor, sentence, intensity, label, split; no absolute paths).
2. Label map order (anger: 0, disgust: 1, fear: 2, happy: 3, neutral: 4, sad: 5).
3. Zero actor leakage across train, val, and test splits.
4. Per-split class counts and balanced distribution.
5. 5-fold GroupKFold cross-validation partitioning with zero actor leakage across folds.
6. Long-tailed train-only manifests (rho=10, 20, 50) with val/test unchanged and imbalance ratio within 5%.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

EXPECTED_CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EXPECTED_LABEL_MAP = {c: i for i, c in enumerate(EXPECTED_CLASSES)}


@pytest.fixture(scope="module")
def manifest_df():
    manifest_path = DATA_DIR / "cremad_manifest.csv"
    assert manifest_path.exists(), f"Manifest file missing: {manifest_path}"
    return pd.read_csv(manifest_path)


@pytest.fixture(scope="module")
def label_map():
    label_map_path = DATA_DIR / "label_map.json"
    assert label_map_path.exists(), f"Label map file missing: {label_map_path}"
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def folds_df():
    folds_path = DATA_DIR / "cremad_folds.csv"
    assert folds_path.exists(), f"Folds file missing: {folds_path}"
    return pd.read_csv(folds_path)


class TestCremadManifest:
    def test_manifest_schema(self, manifest_df):
        expected_cols = ["id", "actor", "sentence", "intensity", "label", "split"]
        assert list(manifest_df.columns) == expected_cols, f"Columns mismatch: {manifest_df.columns}"
        assert len(manifest_df) == 7442, f"Expected 7442 rows, got {len(manifest_df)}"
        assert set(manifest_df["split"].unique()) == {"train", "val", "test"}

    def test_no_absolute_paths(self, manifest_df):
        for col in manifest_df.columns:
            for val in manifest_df[col].astype(str):
                assert not val.startswith(("/", "\\", "C:", "D:", "c:", "d:")), (
                    f"Absolute path detected in column '{col}': {val}"
                )

    def test_label_map_fixed_order(self, label_map):
        assert label_map == EXPECTED_LABEL_MAP, f"Label map mismatch: {label_map}"
        assert list(label_map.keys()) == EXPECTED_CLASSES
        for i, c in enumerate(EXPECTED_CLASSES):
            assert label_map[c] == i

    def test_no_actor_leakage(self, manifest_df):
        train_actors = set(manifest_df[manifest_df["split"] == "train"]["actor"])
        val_actors = set(manifest_df[manifest_df["split"] == "val"]["actor"])
        test_actors = set(manifest_df[manifest_df["split"] == "test"]["actor"])

        # Strict disjoint set assertion
        assert len(train_actors & val_actors) == 0, f"Actor leakage (train & val): {train_actors & val_actors}"
        assert len(train_actors & test_actors) == 0, f"Actor leakage (train & test): {train_actors & test_actors}"
        assert len(val_actors & test_actors) == 0, f"Actor leakage (val & test): {val_actors & test_actors}"

        # Total unique actors
        all_actors = train_actors | val_actors | test_actors
        assert len(all_actors) == 91, f"Expected 91 unique actors, found {len(all_actors)}"
        assert len(train_actors) == 64
        assert len(val_actors) == 14
        assert len(test_actors) == 13

    def test_per_split_class_counts(self, manifest_df):
        train_sub = manifest_df[manifest_df["split"] == "train"]
        val_sub = manifest_df[manifest_df["split"] == "val"]
        test_sub = manifest_df[manifest_df["split"] == "test"]

        assert len(train_sub) == 5229
        assert len(val_sub) == 1148
        assert len(test_sub) == 1065

        # Check emotion distribution in train
        tr_counts = train_sub["label"].value_counts().to_dict()
        for emo in ["anger", "disgust", "fear", "happy", "sad"]:
            assert tr_counts[emo] == 893
        assert tr_counts["neutral"] == 764

        # Check emotion distribution in val
        va_counts = val_sub["label"].value_counts().to_dict()
        for emo in ["anger", "disgust", "fear", "happy", "sad"]:
            assert va_counts[emo] == 196
        assert va_counts["neutral"] == 168

        # Check emotion distribution in test
        te_counts = test_sub["label"].value_counts().to_dict()
        for emo in ["anger", "disgust", "fear", "happy", "sad"]:
            assert te_counts[emo] == 182
        assert te_counts["neutral"] == 155


class TestCremadFolds:
    def test_folds_schema_and_splits(self, folds_df):
        assert "fold" in folds_df.columns
        assert len(folds_df) == 7442

        # Test set rows must have fold == -1
        test_folds = folds_df[folds_df["split"] == "test"]["fold"].unique()
        assert list(test_folds) == [-1], f"Test rows must have fold -1, got: {test_folds}"

        # Train+Val rows must have fold in 0..4
        train_val_df = folds_df[folds_df["split"].isin(["train", "val"])]
        assert set(train_val_df["fold"].unique()) == {0, 1, 2, 3, 4}

    def test_no_actor_leakage_across_folds(self, folds_df):
        train_val_df = folds_df[folds_df["split"].isin(["train", "val"])]
        for f1 in range(5):
            for f2 in range(f1 + 1, 5):
                a1 = set(train_val_df[train_val_df["fold"] == f1]["actor"])
                a2 = set(train_val_df[train_val_df["fold"] == f2]["actor"])
                assert len(a1 & a2) == 0, f"Actor leakage between fold {f1} and {f2}: {a1 & a2}"


class TestLongTailedManifests:
    @pytest.mark.parametrize("rho", [10, 20, 50])
    def test_long_tail_manifest(self, manifest_df, rho):
        lt_path = DATA_DIR / f"cremad_lt_rho{rho}.csv"
        assert lt_path.exists(), f"Long-tailed manifest missing: {lt_path}"
        lt_df = pd.read_csv(lt_path)

        # 1. Val and test splits must remain completely unchanged
        for sp in ["val", "test"]:
            orig_sub = manifest_df[manifest_df["split"] == sp].sort_values(by="id").reset_index(drop=True)
            lt_sub = lt_df[lt_df["split"] == sp].sort_values(by="id").reset_index(drop=True)
            pd.testing.assert_frame_equal(orig_sub, lt_sub, check_like=True)

        # 2. No actor leakage across splits in LT manifest
        train_actors = set(lt_df[lt_df["split"] == "train"]["actor"])
        val_actors = set(lt_df[lt_df["split"] == "val"]["actor"])
        test_actors = set(lt_df[lt_df["split"] == "test"]["actor"])
        assert len(train_actors & val_actors) == 0
        assert len(train_actors & test_actors) == 0
        assert len(val_actors & test_actors) == 0

        # 3. Check imbalance ratio n_0 / n_{C-1} within 5% of target rho
        train_df = lt_df[lt_df["split"] == "train"]
        n_0 = len(train_df[train_df["label"] == EXPECTED_CLASSES[0]])
        n_last = len(train_df[train_df["label"] == EXPECTED_CLASSES[-1]])
        actual_rho = n_0 / n_last
        rel_diff = abs(actual_rho - rho) / rho

        assert rel_diff < 0.05, (
            f"For rho={rho}, expected ratio ~{rho} (+-5%), got {actual_rho:.4f} (rel diff: {rel_diff*100:.2f}%)"
        )

