# Food-101 Imbalanced Dataset Pipeline (food_101_imbalanced_making)

This directory contains all tools, scripts, and unit tests used to construct and verify the **Food-101 Long-Tailed (Food-101-LT)** benchmark for the RCS (Rebalancing with Calibrated Sub-classes) paper pipeline.

---

## File Contents

| File | Purpose |
| :--- | :--- |
| `profile.py` | Mathematical profile generator (`make_counts`) and 6-stage theoretical verification suite (`check_profile`). Supports presets: `mild` (IR=10), `main` (IR=17.5), `stress` (IR=35). |
| `build_split.py` | Seeds class rank permutation (Label 0 = largest class, Label 100 = smallest class), extracts 50 balanced VAL per class, samples `counts[rank]` for TRAIN, preserves 250 untouched balanced TEST per class. Saves `manifest.csv`, `class_map.json`, `config.json`. |
| `verify.py` | Comprehensive verification script: reloads tensors and evaluation folders, asserts shapes, value ranges $[-1, 1]$, exact bincount matching, non-increasing ordering, balanced val/test sets, zero file overlap across splits, prints statistical metrics at $\eta=7$ and $\eta=4$, and saves diagnostic plots to `report/`. |
| `test_food101_lt.py` | Complete automated unit test suite verifying mathematical profiles, boundary assertions, stress preset overrides, and non-zero exit handling on assertion failure. |
| `FOOD101_IMBALANCED_RESEARCH.md` | Academic research technical report detailing mathematical formulations, RCS GMM preconditions, empirical proofs, and statistical metrics. |

---

## Quick Start & Reproduction

### 1. Check Profile
```bash
python food_101_imbalanced_making/profile.py --preset main
```

### 2. Build Split Manifests
```bash
python food_101_imbalanced_making/build_split.py --food101_root ./food_101 --output_dir ./data/food101_lt --preset main
```

### 3. Verify Dataset & Generate Plots
```bash
python food_101_imbalanced_making/verify.py --data_dir ./data/food101_lt --report_dir ./data/food101_lt/report
```

### 4. Run All Unit Tests
```bash
python food_101_imbalanced_making/test_food101_lt.py
```

