#!/usr/bin/env bash
# ==============================================================================
# run_all_cremad.sh
# ==============================================================================
# End-to-end execution script for adapting "multimodel_class_imbalance"
# to CREMA-D (Audio + Video, 6 emotions).
#
# Execution Pipeline:
#   1. Verify raw data integrity and manifest splits (Step 2 & 3)
#   2. Preprocess video face crops (Step 5)
#   3. Extract WavLM audio features and select winning layer (Step 6)
#   4. Extract DINOv2 video features and build feature sets (Step 7)
#   5. Tune Regularized Gated Fusion V2 on train features (Step 8)
#   6. Extract latents, tune RCS, and evaluate all baselines (Step 9)
#   7. Verify zero-leakage safeguards (Step 10)
#   8. Aggregate results, plot visualizations, and compile REPORT.md (Step 10)
#   9. Run full regression test suite (114+ tests)
# ==============================================================================

set -euo pipefail

MEDIA_ROOT="${1:-CREMAD}"

echo "========================================================================"
echo "STARTING CREMA-D END-TO-END MULTIMODAL PIPELINE"
echo "Media Root: ${MEDIA_ROOT}"
echo "========================================================================"

# Step 2: Verify Raw Media Files
echo ""
echo "[1/9] Verifying raw audio and video clips..."
python scripts/verify_cremad_data.py --media_root "${MEDIA_ROOT}"

# Step 5: Video Face-Crop Preprocessing
echo ""
echo "[2/9] Running video face-crop preprocessing..."
python scripts/preprocess_cremad_video.py --media_root "${MEDIA_ROOT}"

# Step 6: Audio Feature Extraction (WavLM-base-plus)
echo ""
echo "[3/9] Extracting WavLM audio representations..."
python scripts/extract_cremad_audio.py --media_root "${MEDIA_ROOT}"

# Step 7: Video Feature Extraction (DINOv2) & Feature Set Building
echo ""
echo "[4/9] Extracting DINOv2 video representations and probing..."
python scripts/extract_cremad_video.py --media_root "${MEDIA_ROOT}"
python scripts/probe_features.py
python scripts/build_cremad_feature_sets.py

# Step 8: Tune Fusion V2 Hyperparameters (Validation Only)
echo ""
echo "[5/9] Tuning Fusion V2 on CREMA-D validation split..."
python scripts/tune_fusion_cremad.py

# Step 9: Latents Extraction, RCS Grid Tuning, and Baselines Benchmark
echo ""
echo "[6/9] Benchmarking rebalancing strategies across balanced and long-tailed settings..."
python scripts/run_cremad_experiments.py --settings balanced lt_rho10 lt_rho20 lt_rho50 --seeds 42 43 44

# Step 10a: Zero-Leakage QA Verification
echo ""
echo "[7/9] Running Leakage QA Verification..."
python scripts/verify_leakage.py

# Step 10b: Results Aggregation and Report Compilation
echo ""
echo "[8/9] Aggregating results and generating results_cremad/REPORT.md..."
python scripts/aggregate_cremad.py

# Step 10c: Full Regression Test Suite
echo ""
echo "[9/9] Running complete regression test suite..."
pytest tests/ -v

echo ""
echo "========================================================================"
echo "CREMA-D PIPELINE COMPLETED SUCCESSFULLY!"
echo "Master Report: results_cremad/REPORT.md"
echo "========================================================================"

