# Pipeline Execution Plan & Dataset Adaptation Guide

This document provides a complete, step-by-step walkthrough for running the codebase, reproducing the winning multimodal emotion recognition benchmarks, and adapting the pipeline to any new multimodal dataset.

---

## 1. System Overview & Architecture

The system implements a state-of-the-art framework for **Multimodal Emotion Recognition under Extreme Class Imbalance**:

```
[Raw Multimodal Data] 
       │ (Text, Video, Audio)
       ▼
1. Feature Extraction (RoBERTa 768-d + VideoMAE 768-d)
       │
       ▼
2. Regularized Gated Multimodal Fusion (Fusion V2)
       │ • Modality Dropout (p=0.15)
       │ • Highway Gate Logits with Softmax Renormalization
       │ • Multi-Task Auxiliary Heads (Text + Video)
       │ • Decoupled Supervised Contrastive Loss (Decoupled SupCon)
       ▼
3. Joint Latent Representation (Z \in R^{512})
       │
       ▼
4. Latent RCS-Optimal Rebalancing
       │ • 64-d PCA Subspace Reduction (eliminates N << D^2 singularity)
       │ • Regularized Diagonal GMM Partitioning on Majority Classes
       │ • Calibrated k-NN (k=5) Covariance Synthesis on Minority Classes
       │ • Dynamic Target Capping (2,500 samples/class)
       ▼
5. MLP Classification Head & Pure Test Evaluation
       └─► Macro-F1: 0.4092 ± 0.0007 | Bal-Acc: 41.78% | G-Mean: 0.3700 (Rank #1 Champion)
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13)
- CUDA-compatible GPU (recommended; tested on NVIDIA RTX 3050 Laptop GPU / A100 / V100)
- Operating System: Linux, macOS, or Windows

### Step 1: Clone Repository
```bash
git clone https://github.com/Charantej06/multimodel_class_imbalance.git
cd multimodel_class_imbalance
```

### Step 2: Create and Activate Virtual Environment
```bash
# On Linux / macOS:
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Workflow A: Instant Reproduction (< 2 Minutes)

If you just cloned the repository and want to immediately verify the winning benchmark without waiting to download raw gigabyte-scale datasets:

### Run RCS-Optimal on Pre-extracted Latents:
```bash
python run_final_rcs.py
```
*or directly from the standalone package:*
```bash
python final_rcs_package/run_final_rcs.py
```

### What happens:
1. Automatically loads the precomputed 512-d joint representations from `final_rcs_package/latents/`.
2. Applies **RCS-Optimal** subspace synthesis across seeds $\{42, 43, 44\}$.
3. Trains the MLP classification head on balanced latents using validation Macro-F1 model selection.
4. Outputs the exact verified benchmark:
   - **Test Macro-F1:** `0.4092 ± 0.0007` (Peak: `0.4101`)
   - **Balanced Accuracy:** `41.78 ± 0.76%` (Peak: `42.43%`)
   - **Geometric Mean:** `0.3700 ± 0.0199`
   - **Raw Accuracy:** `57.89%`

### Inspect Pre-trained Weights & Full Parameter Bundle:
```python
import pickle

with open("final_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

print("RCS Parameters:", bundle["hyperparameters"]["rcs_optimal"])
print("Fusion Loss:", bundle["hyperparameters"]["fusion_training_loss"])
print("Model Weights:", list(bundle["model_weights"].keys()))
print("Evaluation Metrics:", bundle["evaluation_metrics"])
```

---

## 4. Workflow B: Full Pipeline on MELD from Scratch

To run the complete data extraction and training pipeline from the raw MELD dataset:

### Execution Order:
```
Step 1: prepare_meld.py          ──► Unified CSV Manifest
Step 2: feature_extraction.py    ──► Extracted Text & Video Embeddings (.npz)
Step 3: train_fusion_v2.py       ──► Trained Fusion V2 Checkpoint (.pt)
Step 4: extract_latents.py       ──► 512-d Latent Joint Features (.npz)
Step 5: run_final_rcs.py         ──► Final Rebalanced Classification & Metrics
```

### Command Sequence:

#### Step 1: Prepare Raw MELD Data into a Unified Manifest
Place your raw MELD files in `data/` or `MELD.Raw/`, then run:
```bash
python prepare_meld.py --raw_dir MELD.Raw --output_manifest data/meld_manifest.csv
```

#### Step 2: Extract Pretrained Multimodal Backbones
Extracts RoBERTa-base (768-d) and VideoMAE (768-d) features:
```bash
python feature_extraction.py --manifest data/meld_manifest.csv --cache_root embeddings_v3/
```

#### Step 3: Train Fusion V2 (Regularized Gated Fusion with Decoupled SupCon)
```bash
python train_fusion_v2.py --features_dir embeddings_v3 --output_dir results_v3 --configs V2 --seeds 42 43 44
```
*Saves `best_fusion_v2_model.pt` in `results_v3/` based strictly on validation Macro-F1.*

#### Step 4: Extract 512-d Fused Latents from the Trained Checkpoint
```bash
python extract_latents.py --features_dir embeddings_v3 --checkpoint results_v3/best_fusion_v2_model.pt --output_dir results_rcs/latents
```

#### Step 5: Rebalance Latent Space and Evaluate with RCS-Optimal
```bash
python run_final_rcs.py --latents_dir results_rcs/latents --seeds 42 43 44
```

---

## 5. Workflow C: Adapting the Pipeline to ANY New Dataset

If you are a collaborator or researcher applying this pipeline to a new multimodal dataset (e.g., IEMOCAP, CMU-MOSEI, MuSe, or your own custom dataset), follow this guide:

### Step 1: Format Your Dataset Manifest CSV
Create a single metadata CSV (e.g., `data/my_dataset_manifest.csv`) with the following standard columns:

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `id` | string / int | `utt_001` | Unique identifier for each sample |
| `text` | string | `"I really did not expect this to happen."` | Raw utterance text |
| `video_path` | string / path | `"raw_videos/utt_001.mp4"` | Relative or absolute path to video clip (or audio) |
| `label` | string / int | `"surprise"` (or `0, 1, 2...`) | Emotion / class label |
| `split` | string | `"train"`, `"val"`, or `"test"` | Explicit data split |

> *Note: If your dataset does not have predefined splits, `dataset_adapter.py` can automatically partition your dataset with stratified splitting.*

### Step 2: Use `dataset_adapter.py` for Generic Loading
Our `dataset_adapter.py` is format-agnostic and auto-detects columns:
```python
from dataset_adapter import MultimodalDatasetAdapter

adapter = MultimodalDatasetAdapter(
    metadata_path="data/my_dataset_manifest.csv",
    id_col="id",
    label_col="label",
    split_col="split",
    modality_cols={
        "text": "text",
        "video": "video_path",
    },
)

# Access PyTorch datasets directly:
train_ds = adapter.get_split_dataset("train")
val_ds = adapter.get_split_dataset("val")
test_ds = adapter.get_split_dataset("test")
num_classes = adapter.num_classes
label_map = adapter.label_map
```

### Step 3: Extract Features for Your Dataset
Use `feature_extraction.py` or the adapter to generate unimodal embeddings:
- **Text:** RoBERTa-base, DeBERTa, or ModernBERT (`text_dim = 768`)
- **Video / Vision:** VideoMAE, ViT, or DINOv2 (`video_dim = 768`)
- **Audio (if applicable):** AudioMAE (`audio_dim = 768`)

Save the extracted tensors as `.npz` files:
- `train_features.npz`: contains arrays `X_text`, `X_video`, `y`
- `val_features.npz`: contains arrays `X_text`, `X_video`, `y`
- `test_features.npz`: contains arrays `X_text`, `X_video`, `y`

### Step 4: Train Fusion V2 on Your Dataset
Run the Fusion V2 trainer with your class count:
```bash
python train_fusion_v2.py --features_dir path/to/my_features --output_dir results_my_dataset --configs V2 --seeds 42 43 44
```

### Step 5: Extract 512-d Latents
Extract the joint representations from your trained model:
```bash
python extract_latents.py --features_dir path/to/my_features --checkpoint results_my_dataset/best_fusion_v2_model.pt --output_dir latents_my_dataset
```

### Step 6: Rebalance with RCS-Optimal
Rebalance your new dataset's latent space:
```bash
python run_final_rcs.py --latents_dir latents_my_dataset --pca_dim 64 --covariance_type diag --seeds 42 43 44
```

---

## 6. Hyperparameter Tuning Guide for New Datasets

When moving to a new dataset with different sample counts or imbalance ratios, use these heuristics to configure RCS:

| Hyperparameter | Flag | Default | Recommended Adjustment for New Datasets |
| :--- | :--- | :--- | :--- |
| **PCA Subspace Dim** | `--pca_dim` | `64` | Set to $\min(64, \frac{N_{\text{smallest\_minority}}}{4})$. In high-dimensional spaces ($D=512$), small classes cannot estimate full covariance matrices. Projecting to $32$ or $64$ ensures numerical stability and prevents matrix singularity. |
| **Covariance Type** | `--covariance_type`| `diag` | Always keep as `'diag'`. Diagonal covariance regularizes component variance and avoids Cholesky decomposition failures. |
| **Target Cap** | `--target_cap` | `2500` | Set to approximately $2 \times$ to $4 \times$ your median class frequency, or cap at $\min(N_{\text{majority}}, 3000)$. Never allow synthetic samples to exceed $85\%$ of the total training points for any single class. |
| **Imbalance Tuner $\eta$** | `--eta` | `7` | Controls which classes are treated as majority vs minority: $\text{threshold} = \text{target} // \eta$. For mild imbalance (ratio < 5:1), set $\eta=3$. For extreme tail imbalance (ratio > 20:1), keep $\eta=7$. |
| **Nearest Neighbors $k$** | `--k` | `5` | Number of neighboring GMM centroids used to interpolate minority covariance. Use $k=3$ if total majority components $< 10$; use $k=5$ otherwise. |

---

## 7. Directory Structure Reference

```
multimodel_class_imbalance/
├── .gitignore                    # Excludes raw data, video clips, and caches
├── README.md                     # GitHub project presentation
├── plan.md                       # This execution and adaptation guide
├── context.md                    # Technical reference, formulas, and hyperparameter tables
├── research.md                   # Research log, theoretical rationale, and literature benchmarks
├── requirements.txt              # Production dependency specifications
├── final_model_bundle.pkl        # Self-contained model pickle (weights + params + metrics)
│
├── prepare_meld.py               # Step 1: Raw dataset preparation manifest generator
├── dataset_adapter.py            # Generic dataset loader (works with ANY dataset)
├── feature_extraction.py         # Step 2: Pretrained backbone extractor (RoBERTa + VideoMAE)
├── train_fusion_v2.py            # Step 3: Multimodal Fusion V2 training with Decoupled SupCon
├── extract_latents.py            # Step 4: 512-d latent extraction from trained Fusion V2
├── rcs_engine.py                 # Core SOTA RCS-Optimal latent rebalancing engine
├── run_final_rcs.py              # Step 5: End-to-end RCS-Optimal evaluation runner
│
├── encoders/
│   ├── __init__.py
│   └── registry.py               # Model registry for feature backbones and pooling
│
├── models/
│   ├── __init__.py
│   └── regularized_gated_fusion_v2.py  # Best Fusion V2 architecture
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset_adapter.py
│   ├── test_feature_extraction.py
│   ├── test_meld_integration.py
│   ├── test_registry.py
│   └── test_regularized_gated_fusion_v2.py
│
└── final_rcs_package/            # Standalone package for instant reproduction
    ├── README.md
    ├── fusion_model.py
    ├── rcs_engine.py
    ├── run_final_rcs.py
    ├── final_model_bundle.pkl
    ├── latents/
    │   ├── best_fusion_v2_model.pt
    │   ├── latents_train.npz
    │   ├── latents_val.npz
    │   └── latents_test.npz
    └── results/
        ├── rcs_championship_metrics.json
        ├── benchmark_all_methods.json
        └── all_fusion_methods_comparison_grid.png
```

---

## 8. Troubleshooting & Best Practices

1. **GPU Out of Memory (OOM):**
   - The codebase defaults to `torch.cuda.set_per_process_memory_fraction(0.80, device=0)`.
   - If running on smaller GPUs (e.g. 4GB VRAM), adjust batch size:
     ```bash
     python train_fusion_v2.py --batch_size 128
     ```
2. **Ill-Conditioned Covariance Matrices:**
   - If adding custom GMM components on very small classes, ensure `regularization_covar=1e-3` and `ensure_psd` are active. `rcs_engine.py` handles this automatically.
3. **Reproducibility:**
   - Always run evaluations over seeds $\{42, 43, 44\}$ and report $\text{Mean} \pm \text{Std}$.
   - Model selection is performed strictly on validation Macro-F1; test set metrics are evaluated only once at the selected epoch.
