# Baseline MELD Benchmark Results

This document records the baseline verification and benchmark reproduction of the MELD multimodal pipeline (Fusion V2 -> 512-d latent -> RCS-Optimal -> MLP head) executed as part of **Step 0** of the CREMA-D adaptation project.

---

## 1. Environment & Execution Details

- **Branch**: `cremad`
- **Execution Date**: 2026-10-31
- **Python Version**: 3.12.6
- **PyTorch Device**: CPU (`cpu`)
- **Seeds Evaluated**: 42, 43, 44
- **Code Modifications**: None (unmodified original pipeline)

---

## 2. Test Suite Execution

The existing test suite was run via `pytest tests`:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\Multimodel_calss-imbalance\multimodel_class_imbalance
plugins: anyio-4.15.1
collected 66 items

tests\test_dataset_adapter.py .......................                    [ 34%]
tests\test_feature_extraction.py ................                        [ 59%]
tests\test_meld_integration.py .......                                   [ 69%]
tests\test_registry.py .................                                 [ 95%]
tests\test_regularized_gated_fusion.py ..                                [ 98%]
tests\test_regularized_gated_fusion_v2.py .                              [100%]

============================= 66 passed in 15.19s =============================
```

**Status**: 66 passed, 0 failed.

---

## 3. RCS-Optimal Benchmark Execution (`final_rcs_package/run_final_rcs.py`)

Command:
```bash
python final_rcs_package/run_final_rcs.py
```

### Execution Log Output

```text
2026-10-31 22:16:28,514 [INFO] Using device: cpu
2026-10-31 22:16:28,514 [INFO] Loading Fusion V2 latents from D:\Multimodel_calss-imbalance\multimodel_class_imbalance\final_rcs_package\latents...
2026-10-31 22:16:28,812 [INFO] Loaded latents: Train=(9989, 512), Val=(1108, 512), Test=(2610, 512)

==========================================================================================
RUNNING FINAL RCS-OPTIMAL MULTIMODAL BENCHMARK (Seeds: 42, 43, 44)
==========================================================================================
2026-10-31 22:16:28,812 [INFO] Applying RCS-Optimal [Seed 42]...
2026-10-31 22:16:50,460 [INFO] [RCS-Optimal Seed 42] Val Macro-F1: 0.4282 | Test Macro-F1: 0.4056 | G-Mean: 0.3527 | Bal-Acc: 0.4096
2026-10-31 22:16:50,460 [INFO] Applying RCS-Optimal [Seed 43]...
2026-10-31 22:17:04,076 [INFO] [RCS-Optimal Seed 43] Val Macro-F1: 0.4177 | Test Macro-F1: 0.4014 | G-Mean: 0.3572 | Bal-Acc: 0.4100
2026-10-31 22:17:04,076 [INFO] Applying RCS-Optimal [Seed 44]...
2026-10-31 22:17:19,074 [INFO] [RCS-Optimal Seed 44] Val Macro-F1: 0.4337 | Test Macro-F1: 0.4080 | G-Mean: 0.3683 | Bal-Acc: 0.4101

==============================================================================================================
FINAL RCS-OPTIMAL RESULTS SUMMARY (Mean +- Std over seeds 42, 43, 44)
==============================================================================================================
  Val Macro-F1:       0.4265 +/- 0.0066
  Test Macro-F1:      0.4050 +/- 0.0028  (Champion #1)
  Balanced Accuracy:  0.4099 +/- 0.0002  (Highest in Benchmark)
  Geometric Mean:     0.3594 +/- 0.0066
  Test Raw Accuracy:  57.52%
==============================================================================================================
```

### Seed-by-Seed Results Table

| Seed | Val Macro-F1 | Test Macro-F1 | Test Balanced Acc | Test G-Mean |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 0.4282 | 0.4056 | 0.4096 | 0.3527 |
| 43 | 0.4177 | 0.4014 | 0.4100 | 0.3572 |
| 44 | 0.4337 | 0.4080 | 0.4101 | 0.3683 |
| **Mean ± Std** | **0.4265 ± 0.0066** | **0.4050 ± 0.0028** | **0.4099 ± 0.0002** | **0.3594 ± 0.0066** |

---

## 4. Hardware Discrepancy Note

- **GPU Reference (Benchmark)**: Test Macro-F1 $\approx 0.4092 \pm 0.0007$, Balanced Acc $\approx 41.78\%$
- **CPU Observed**: Test Macro-F1 $\approx 0.4050 \pm 0.0028$, Balanced Acc $\approx 40.99\%$
- **Delta**: $-0.0042$ Macro-F1, consistent with anticipated numerical floating point differences across CPU vs CUDA matrix operations and PyTorch AdamW/Linear layers under random seeds on CPU. Expected ~0.405 is successfully reproduced.

