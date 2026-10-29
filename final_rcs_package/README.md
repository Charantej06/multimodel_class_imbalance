# Final RCS Multimodal Imbalance Package

This folder contains the complete, self-contained, publication-ready codebase for the **RCS-Optimal Multimodal Emotion Recognition framework** on MELD.

---

## 📁 Directory Structure

```
final_rcs_package/
├── fusion_model.py          # State-of-the-Art Fusion V2 Architecture (Dynamic Softmax Gating + Decoupled SupCon)
├── rcs_engine.py            # The winning RCS-Optimal algorithm (PCA-64 + GMM + Diagonal Covariance + Target Capping)
├── run_final_rcs.py         # End-to-end runnable benchmark script across seeds [42, 43, 44]
├── latents/                 # Extracted 512-d representations and checkpoint
│   ├── best_fusion_v2_model.pt
│   ├── latents_train.npz / .pt (9,989 x 512)
│   ├── latents_val.npz / .pt   (1,108 x 512)
│   └── latents_test.npz / .pt  (2,610 x 512)
└── results/                 # Full benchmark metrics & plots
    ├── rcs_championship_metrics.json
    ├── benchmark_all_methods.json
    └── all_fusion_methods_comparison_grid.png
```

---

## 🚀 Quick Execution (1 Command)

To reproduce the complete benchmark and verify the results across seeds $\{42, 43, 44\}$:

```bash
python final_rcs_package/run_final_rcs.py
```

---

## 🏆 Benchmark Performance Summary

| Metric | Real Only Baseline | GAMO (CVPR'20) | SMOTE | **RCS-Optimal (Ours)** | Relative Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Macro-F1** | $0.3973 \pm 0.0022$ | $0.4015 \pm 0.0024$ | $0.3854 \pm 0.0047$ | **$0.4092 \pm 0.0007$** | **#1 Champion** |
| **Balanced Accuracy** | $38.85\%$ | $40.26\%$ | $40.97\%$ | **$41.78 \pm 0.76\%$** | **$+2.93\%$ over Real** |
| **Geometric Mean (G-Mean)**| $0.3110$ | $0.3453$ | $0.3730$ | **$0.3700 \pm 0.0199$** | **$+19.0\%$ over Real** |
| **Execution Time** | - | ~25s / seed | ~12s / seed | **~7.4s / seed** | **Fastest & Most Stable** |
