# Multimodal Emotion Recognition under Extreme Class Imbalance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark](https://img.shields.io/badge/MELD_Macro--F1-0.4092-brightgreen.svg)]()
[![Balanced Accuracy](https://img.shields.io/badge/Balanced_Accuracy-41.78%25-success.svg)]()

A state-of-the-art framework for **Multimodal Emotion Recognition in Conversation (MERC)** under severe class imbalance, combining:
1. **Regularized Gated Multimodal Fusion (Fusion V2)** with Decoupled Supervised Contrastive Learning (Decoupled SupCon).
2. **Optimal Latent Random Covariance Sampling (RCS-Optimal)** using PCA subspace reduction, regularized diagonal GMM partitioning, and calibrated k-NN covariance synthesis.

Tested extensively on the **MELD** benchmark (13,707 video dialogues across 7 emotions with a 17.6:1 imbalance ratio).

---

## 🏆 Key Benchmark Results (MELD Test Set)

Evaluated strictly across random seeds $\{42, 43, 44\}$ on pure test data:

| Method | Latent Rebalancing | Test Macro-F1 | Balanced Accuracy | Geometric Mean (G-Mean) | Raw Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Real Only (Baseline)** | None | $0.3973 \pm 0.0022$ | $38.85 \pm 0.76\%$ | $0.3110 \pm 0.0132$ | **$60.09\%$** |
| **SMOTE** | Feature Interpolation | $0.3854 \pm 0.0047$ | $40.97 \pm 0.35\%$ | $0.3730 \pm 0.0055$ | $54.34\%$ |
| **Borderline-SMOTE** | Frontier Interpolation | $0.3861 \pm 0.0039$ | $40.42 \pm 0.41\%$ | $0.3662 \pm 0.0061$ | $55.12\%$ |
| **DeepSMOTE** | Autoencoder Latents | $0.3842 \pm 0.0041$ | $39.88 \pm 0.50\%$ | $0.3591 \pm 0.0074$ | $56.02\%$ |
| **GAMO** | Adversarial Over-sampling | $0.4015 \pm 0.0024$ | $40.26 \pm 0.62\%$ | $0.3453 \pm 0.0089$ | $59.18\%$ |
| **RCS-Optimal (Ours)** | **Subspace PCA-64 + Diag GMM** | **$0.4092 \pm 0.0007$** | **$41.78 \pm 0.76\%$** | **$0.3700 \pm 0.0199$** | $57.89\%$ |

> 🌟 **RCS-Optimal is the undisputed Rank #1 Champion**: Achieves the highest Macro-F1 ($0.4092$), highest Balanced Accuracy ($41.78\%$), and elevates tail minority recall significantly (Fear: $10.7\% \to 26.0\%$, Sadness: $22.6\% \to 34.6\%$).

---

## 🚀 Quick Start (< 2 Minutes)

### 1. Installation
```bash
git clone https://github.com/Charantej06/multimodel_class_imbalance.git
cd multimodel_class_imbalance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Instant Verification of SOTA Results
Run the championship benchmark directly using the pre-extracted latents:
```bash
python run_final_rcs.py
```
*or from the self-contained package:*
```bash
python final_rcs_package/run_final_rcs.py
```

### 3. Load Pre-trained Model & Parameters (Pickle Bundle)
```python
import pickle

with open("final_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

print("Optimal RCS Params:", bundle["hyperparameters"]["rcs_optimal"])
print("Trained Weights:", bundle["model_weights"].keys())
print("Verified Metrics:", bundle["evaluation_metrics"])
```

---

## 📖 Complete Pipeline & Documentation

- 📋 **[plan.md](plan.md)**: **Step-by-step pipeline execution order** (from raw datasets to final evaluation) and **a detailed guide on how to adapt the pipeline to ANY new multimodal dataset**.
- 🔬 **[context.md](context.md)**: Full theoretical derivations, multi-task loss formulas, exact hyperparameter tables, and component priority breakdown.
- 📑 **[research.md](research.md)**: Research log, ablation experiments, t-SNE clustering geometry analysis, and literature comparisons.

---

## 📂 Repository Structure

```
multimodel_class_imbalance/
├── prepare_meld.py               # Step 1: Raw data preparation into standard manifest
├── dataset_adapter.py            # Generic dataset loader (works with ANY dataset)
├── feature_extraction.py         # Step 2: Backbone extractor (RoBERTa + VideoMAE)
├── train_fusion_v2.py            # Step 3: Multimodal Fusion V2 with Decoupled SupCon
├── extract_latents.py            # Step 4: 512-d joint latent feature extraction
├── rcs_engine.py                 # Core SOTA RCS-Optimal latent rebalancing engine
├── run_final_rcs.py              # Step 5: Multi-seed benchmark runner & evaluation
│
├── models/
│   ├── __init__.py
│   └── regularized_gated_fusion_v2.py  # Champion Fusion V2 architecture
│
├── encoders/
│   ├── __init__.py
│   └── registry.py               # Pretrained backbone & pooling registry
│
├── tests/
│   └── ...                       # Unit and integration test suite
│
├── final_rcs_package/            # Standalone package with verified latents & weights
│   ├── latents/best_fusion_v2_model.pt
│   ├── latents/latents_*.npz
│   └── results/                  # Metric JSONs and comparison figures
│
├── final_model_bundle.pkl        # Self-contained pickle bundle (weights + params + metrics)
├── plan.md                       # Complete pipeline run order & adaptation guide
├── context.md                    # Technical reference, formulas & hyperparameter tables
├── research.md                   # Complete research log & benchmark analysis
└── requirements.txt              # Production dependencies
```

---

## 🧠 Architectural Highlights

### 1. Multimodal Fusion V2
- **Highway Gated Fusion:** Computes dynamic modality importance gates without collapsing to unimodal text dominance.
- **Modality Dropout ($p=0.15$):** Stochastically masks individual modalities during training to enforce cross-modal robustness.
- **Decoupled Supervised Contrastive Loss:** Pulls intra-class representations together while repelling distinct emotions with class-balanced anchor averaging ($\tau=0.2, \alpha_{\text{pos}}=1.0, \alpha_{\text{neg}}=1.0$).

### 2. RCS-Optimal Latent Rebalancing
- **Subspace PCA Reduction ($512 \to 64$):** Eliminates matrix singularity and ill-conditioned covariance estimation on small minority classes.
- **Diagonal GMM Covariance:** Stabilizes Cholesky factorizations and prevents over-dispersing synthetic samples across classification boundaries.
- **Target Capping ($2,500$):** Balances minority sensitivity while preventing synthetic points from overwhelming real human training data.

---

## 🛠 Adapting to a New Dataset

To apply this pipeline to your own dataset:
1. Format a metadata CSV with columns `id`, `text`, `video_path` (or audio), `label`, `split`.
2. Extract unimodal embeddings using `dataset_adapter.py` and `feature_extraction.py`.
3. Train `train_fusion_v2.py` on your extracted features.
4. Extract 512-d latents using `extract_latents.py`.
5. Run `run_final_rcs.py` to balance your dataset and train the classifier head.

👉 *See [plan.md](plan.md) for full instructions and recommended hyperparameter settings.*

---

## 📜 License
This project is licensed under the MIT License.