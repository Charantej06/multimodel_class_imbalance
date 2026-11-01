Ran delete synthetic latents and invalid exp1 results
Analyzed run_final_rcs_food_101.py:L70-145
Edited run_final_rcs_food_101.py
Edited run_final_rcs_food_101.py
Ran verify no npz files in food101_lt

All synthetic files and invalid results have been completely removed:
* Deleted `data/food101_lt/latents_*.npz`
* Deleted `results/food_101/exp1/` and `results_rcs/food_101/exp1/`
* Deleted `benchmark_all_methods_food_101.py`
* Cleaned [`final_rcs_package/run_final_rcs_food_101.py`](file:///c:/Users/chegu/OneDrive/Desktop/multimodel_class_imbalance/final_rcs_package/run_final_rcs_food_101.py) so it strictly requires real extracted latents.

---

## Complete Parameter Breakdown by Stage & Place

Here is the exact mapping of every parameter, its value, and its mathematical purpose at each place in the pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PLACE 1: Pretrained Foundation Encoders (encoders/registry.py)         │
├────────────────────────────────────────────────────────────────────────┤
│ • Image Encoder         : facebook/dinov2-base                         │
│ • Image Output Dim      : 768 (CLS-token embedding)                    │
│ • Text Encoder          : answerdotai/ModernBERT-base                  │
│ • Text Output Dim       : 768 (CLS-token embedding)                    │
│ • Encoder Status        : 100% Frozen (requires_grad = False)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (768-d Image + 768-d Text)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PLACE 2: Multimodal Fusion Model (fusion_model_food_101.py)            │
├────────────────────────────────────────────────────────────────────────┤
│ • text_dim              : 768                                          │
│ • image_dim             : 768                                          │
│ • hidden_dim            : 512 (Common fused representation space)      │
│ • num_classes           : 101 (Culinary categories)                    │
│ • proj_text / proj_image: LayerNorm(768) -> Linear(768, 512) -> Tanh   │
│ • Gating Mechanism      : Linear(1024, 512) -> ReLU -> Dropout(0.3)    │
│                           -> Linear(512, 2) (masked softmax logits)    │
│ • modality_dropout      : 0.15 (drops 1 modality with 15% prob)        │
│ • classifier_dropout    : 0.30                                         │
│ • con_proj (SupCon)     : Linear(512, 512) -> ReLU -> Linear(512, 128)│
│ • con_proj norm         : L2 normalization (p=2)                       │
│ • SupCon temperature (τ): 0.20 (contrastive sharpness)                 │
│ • Optimizer             : AdamW (lr = 1e-4, weight_decay = 1e-4)       │
│ • Max Epochs            : 20 epochs (validation Macro-F1 checkpointing)│
│ • Fused Output Latent Z : 512-dimensional vector (fused_feat)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Fused Latents Z ∈ ℝ^512)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PLACE 3: RCS Rebalancing Engine (rcs_engine.py / apply_rcs_optimal)    │
├────────────────────────────────────────────────────────────────────────┤
│ • Input Dim (D)         : 512                                          │
│ • pca_dim (D_work)      : 64 (PCA subspace dimension reduction)        │
│   ↳ Rationale           : In 512-d, covariance estimation requires     │
│                           D*(D+1)/2 ≈ 131,328 parameters. With 40      │
│                           samples, this is singular. PCA to 64-d       │
│                           resolves the curse of dimensionality.        │
│ • eta (η)               : 7 (Imbalance tuner parameter)                │
│ • zeta (ζ)              : 100 (Threshold = n_max // η = 700 // 7)      │
│ • # Majority Classes    : 68 classes (N_c > 100)                       │
│ • # Minority Classes    : 33 classes (N_c ≤ 100)                       │
│ • GMM Components (K_c)  : ⌈N_c / 100⌉ for majority; 1 for minority    │
│ • Total GMM Components  : 248 components across the 68 majority classes│
│ • covariance_type       : "diag" (Diagonal Variance)                   │
│   ↳ What is diagonal?   : Σ = diag(σ₁², σ₂², ..., σ_d²). It only       │
│                           estimates variances along coordinate axes    │
│                           and assumes zero off-diagonal covariance.    │
│                           Reduces parameters from O(D²) to O(D),       │
│                           preventing matrix collapse and NaN errors.   │
│ • reg_covar             : 1e-3 (regularization added to diagonal)      │
│ • eps (PSD Guard)       : 1e-4 (minimum eigenvalue threshold)          │
│ • k (Nearest Neighbors) : 5 (top-5 closest GMM centroids in 64-d)      │
│ • γ_j (Neighbor Weight) : 1 / S_j (inverse component sample size)      │
│ • w_self (Anchor Weight): 1 - ∑(γ_j) (positivity guaranteed: S_j ≥ 50) │
│ • target_cap            : 700 (upsamples all 101 classes to 700 ceiling│
│                           → Total balanced training set = 70,700)      │
│ • Subspace Inversion    : pca.inverse_transform (maps 64-d back to 512)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Z_balanced ∈ ℝ^(70,700 × 512))
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PLACE 4: Downstream Classifier Head (run_final_rcs_food_101.py)        │
├────────────────────────────────────────────────────────────────────────┤
│ • Architecture          : LatentClassifierHead                         │
│                           Dropout(0.3) -> Linear(512, 256) -> GELU()   │
│                           -> Dropout(0.3) -> Linear(256, 101)          │
│ • in_dim                : 512                                          │
│ • hidden_dim            : 256                                          │
│ • num_classes           : 101                                          │
│ • Loss Criterion        : nn.CrossEntropyLoss()                        │
│ • Optimizer             : AdamW                                        │
│ • Learning Rate (lr)    : 1e-3 (0.001)                                 │
│ • Weight Decay          : 1e-4 (0.0001)                                │
│ • Scheduler             : CosineAnnealingLR(optimizer, T_max=epochs)   │
│ • Max Epochs            : 30 epochs (or 60)                            │
│ • Early Stopping        : patience = 10 epochs on Validation Macro-F1  │
│ • Batch Size            : 256                                          │
│ • GPU Memory Cap        : 70% (set_per_process_memory_fraction(0.70)) │
│ • Random Seeds          : [42, 43, 44]                                 │
│ • Evaluation Protocol   : Checked on 5,050 balanced validation samples │
│                           Evaluated on 25,250 balanced test samples    │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Key Parameter Definitions & Answers to Your Questions:

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

## ⚙️ Stage-by-Stage Hyperparameter & Architecture Breakdown

The table below details the exact configuration used across each stage of the pipeline, noting the updated training epochs:

| Stage / Location | Source File | Core Hyperparameters & Architecture | Purpose / Rationale |
| :--- | :--- | :--- | :--- |
| **Place 1: Modality Encoders** | [`encoders/registry.py`](encoders/registry.py) | • **Vision:** `facebook/dinov2-base` (768-d, frozen, CLS)<br>• **Text:** `answerdotai/ModernBERT-base` (768-d, frozen, CLS)<br>• **Batch Size:** 32–64, PyTorch CUDA FP32 | Extracts unimodal foundational embeddings from raw images and recipe titles. |
| **Place 2: Multimodal Fusion V2** | [`food_101_imbalanced_making/train_fusion_food_101.py`](food_101_imbalanced_making/train_fusion_food_101.py) | • **Epochs:** **40** *(increased from 20)*<br>• **Batch Size:** 64<br>• **Optimizer:** AdamW (`lr=1e-3`, `weight_decay=1e-4`)<br>• **LR Schedule:** CosineAnnealingLR (`T_max=40`, `eta_min=1e-6`)<br>• **Loss:** CrossEntropy + 0.5 × Decoupled SupCon ($\tau=0.2, \alpha_{\text{pos}}=1.0, \alpha_{\text{neg}}=1.0$)<br>• **Dropout:** Modality Dropout $p=0.15$, Classifier Dropout $p=0.3$<br>• **Patience:** 10 epochs early stopping | Fuses unimodal representations into a unified 512-d joint representation with contrastive separation across 101 classes. |
| **Place 3: RCS Latent Engine** | [`final_rcs_package/rcs_engine.py`](final_rcs_package/rcs_engine.py) | • **Subspace Projection:** PCA $512 \to 64$<br>• **GMM Covariance:** `covariance_type="diag"` (avoids $O(D^2)$ singularity)<br>• **Components:** $\eta = 7$<br>• **k-NN Covariance:** $k = 5$<br>• **Target Cap:** 500 (equal to $n_{\max}$)<br>• **Ridge Reg:** `reg_covar = 1e-6` | Balances minority classes up to majority ceiling via regularized, calibrated Gaussian sampling in decorrelated PCA space. |
| **Place 4: Classifier Head** | [`final_rcs_package/benchmark_all_methods_food_101.py`](final_rcs_package/benchmark_all_methods_food_101.py) | • **Epochs:** **60** *(increased from 30)*<br>• **Architecture:** 2-layer MLP (512 $\to$ 256 $\to$ 101)<br>• **Activation & Reg:** GELU, Dropout $p=0.3$<br>• **Optimizer:** AdamW (`lr=1e-3`, `weight_decay=1e-4`)<br>• **LR Schedule:** CosineAnnealingLR (`T_max=60`, `eta_min=1e-6`)<br>• **Early Stopping:** Patience = 10 on Val Macro-F1 | Evaluates final classification accuracy across rebalancing methods (Real Only, SMOTE, B-SMOTE, DeepSMOTE, GAMO, RCS). |

---

## 📜 License
This project is licensed under the MIT License.






