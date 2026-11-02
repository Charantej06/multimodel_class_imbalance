"""
tests/test_leakage_qa.py
========================
Automated regression tests for data leakage safeguards (Step 10).
"""

from scripts.verify_leakage import (
    verify_actor_disjointness,
    verify_scaler_integrity,
    verify_test_isolation_during_tuning,
)


class TestLeakageQA:
    def test_actor_splits_have_zero_overlap(self):
        assert verify_actor_disjointness() is True

    def test_scaler_fit_on_train_only(self):
        assert verify_scaler_integrity() is True

    def test_test_set_not_loaded_during_tuning(self):
        assert verify_test_isolation_during_tuning() is True

