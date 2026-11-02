"""
scripts/plot_tsne_cremad.py
===========================
Step 11: t-SNE 2D visualizations of CREMA-D representations across modalities,
imbalance settings, RCS latent synthesis, actor identity invariance, and error patterns.

Settings:
  - Seed: 42 only
  - Decomposition: PCA to 50 dimensions first
  - t-SNE: sklearn TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
  - Colors: Fixed mapping from data/label_map.json
  - Outputs: Both PNG (300 dpi) and PDF saved to results_cremad/tsne/
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MPLCONFIGDIR"] = str(REPO_ROOT / ".mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from rcs_engine import apply_rcs_optimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("tsne_cremad")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results_cremad"
LATENTS_DIR = RESULTS_DIR / "latents"
OUTPUT_DIR = RESULTS_DIR / "tsne"

EMOTION_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
EMOTION_COLORS = {
    0: "#d62728",  # anger: red
    1: "#2ca02c",  # disgust: green
    2: "#9467bd",  # fear: purple
    3: "#ff7f0e",  # happy: orange
    4: "#1f77b4",  # neutral: blue
    5: "#8c564b",  # sad: brown
}

SETTING_LABELS = {
    "balanced": r"Balanced ($\rho \approx 1.2$)",
    "lt_rho10": r"Long-Tail ($\rho = 10$)",
    "lt_rho20": r"Long-Tail ($\rho = 20$)",
    "lt_rho50": r"Long-Tail ($\rho = 50$)",
}


def run_tsne(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Standardized t-SNE embedding pipeline with PCA preprocessing to 50 dims."""
    n_pca = min(50, X.shape[1])
    X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(X_pca)


def save_figure(fig: plt.Figure, name: str) -> Tuple[Path, Path]:
    """Saves figure in both PNG (300 dpi) and PDF formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", png_path.name, pdf_path.name)
    return png_path, pdf_path


def plot_figure1_settings(y_test: np.ndarray):
    """
    Figure 1 (settings): TEST latents of fusion_v2 for balanced, lt_rho10,
    lt_rho20, lt_rho50 (2x2 grid), colored by true emotion.
    """
    logger.info("Generating Figure 1: Test Latents Across Imbalance Settings...")
    settings = ["balanced", "lt_rho10", "lt_rho20", "lt_rho50"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, s in enumerate(settings):
        ax = axes[idx]
        latent_file = LATENTS_DIR / f"{s}_fusion_v2_seed42.npz"
        latents = np.load(latent_file)
        Z_te = latents["Z_te"]

        emb = run_tsne(Z_te, seed=42)

        for c_idx, c_name in enumerate(EMOTION_NAMES):
            mask = y_test == c_idx
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=EMOTION_COLORS[c_idx],
                label=c_name,
                alpha=0.75,
                s=28,
                edgecolors="none",
            )

        ax.set_title(f"Fusion V2 - {SETTING_LABELS[s]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Put shared legend on the top right axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        frameon=True,
        fontsize=11,
        title="Emotion Class",
        title_fontsize=11,
    )
    fig.suptitle(
        "Figure 1: Fusion V2 Latent Representations on Held-Out Test Split Across Imbalance Regimes (Seed 42)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, "fig1_settings_tsne")


def plot_figure2_modalities(y_test: np.ndarray, test_features: Dict[str, np.ndarray]):
    """
    Figure 2 (modalities): TEST split, 4 panels:
      1. Audio-only features (WavLM, 768-d)
      2. Video-only features (DINOv2, 1536-d)
      3. Concat features (2304-d)
      4. Fusion V2 latent (balanced, 256-d)
    Adds silhouette score on the original (non-t-SNE) space per panel in the title.
    """
    logger.info("Generating Figure 2: Representation Modalities Comparison...")
    X_audio = test_features["X_audio"]
    X_video = test_features["X_video"]
    X_concat = np.concatenate([X_audio, X_video], axis=1)

    latents_balanced = np.load(LATENTS_DIR / "balanced_fusion_v2_seed42.npz")["Z_te"]

    modalities = [
        ("Audio-Only (WavLM Layer 7)", X_audio, 768),
        ("Video-Only (DINOv2 Face)", X_video, 1536),
        ("Concatenated Features", X_concat, 2304),
        ("Fusion V2 Decoupled Latent", latents_balanced, 256),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    axes = axes.flatten()

    for idx, (name, X_orig, dim) in enumerate(modalities):
        ax = axes[idx]
        sil_score = silhouette_score(X_orig, y_test)
        emb = run_tsne(X_orig, seed=42)

        for c_idx, c_name in enumerate(EMOTION_NAMES):
            mask = y_test == c_idx
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=EMOTION_COLORS[c_idx],
                label=c_name,
                alpha=0.75,
                s=28,
                edgecolors="none",
            )

        ax.set_title(
            f"{name} ({dim}-d)\nOriginal Space Silhouette Score: {sil_score:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        frameon=True,
        fontsize=11,
        title="Emotion Class",
        title_fontsize=11,
    )
    fig.suptitle(
        "Figure 2: Representation Geometry Across Modalities (Test Split, Seed 42, with Silhouette Scores)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, "fig2_modalities_tsne")


def plot_figure3_rcs_synthesis():
    """
    Figure 3 (RCS synthesis, lt_rho50 and lt_rho10):
    TRAIN latents, real points as circles and RCS synthetic points as small triangles.
    Fits ONE t-SNE jointly on real+synthetic per setting; shows:
      Column 1: Real points only (from joint embedding)
      Column 2: Real (circles) + Synthetic (triangles)
    Reports exact real and synthetic counts per class in console and titles.
    """
    logger.info("Generating Figure 3: RCS Synthesis Geometry (lt_rho10 & lt_rho50)...")
    settings = ["lt_rho10", "lt_rho50"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for row_idx, s in enumerate(settings):
        lat = np.load(LATENTS_DIR / f"{s}_fusion_v2_seed42.npz")
        Z_tr = lat["Z_tr"]
        feat = np.load(DATA_DIR / f"features_cremad_{s}" / "train_features.npz")
        y_tr = feat["y"]

        # Read best tuned config
        df = pd.read_csv(RESULTS_DIR / f"rcs_grid_{s}.csv")
        best = df.sort_values(by="val_macro_f1", ascending=False).iloc[0]

        min_c = int(np.min(np.bincount(y_tr)))
        p_dim = min(int(best["pca_dim"]), max(1, min_c // 4))
        mss = None if pd.isna(best["max_synthetic_share"]) else float(best["max_synthetic_share"])

        Z_aug, y_aug, info = apply_rcs_optimal(
            Z_train=Z_tr,
            y_train=y_tr,
            label_names=EMOTION_NAMES,
            pca_dim=p_dim,
            covariance_type="diag",
            target_cap=int(best["target_cap"]),
            eta=int(best["eta"]),
            k=int(best["k"]),
            seed=42,
            max_synthetic_share=mss,
        )

        n_real = len(Z_tr)
        n_syn = len(Z_aug) - n_real

        logger.info(
            "%s Synthesis Counts: Real=%d, Synthetic=%d, Total=%d",
            s,
            n_real,
            n_syn,
            len(Z_aug),
        )
        for c_idx, c_name in enumerate(EMOTION_NAMES):
            c_real = int(np.sum(y_tr == c_idx))
            c_total = int(np.sum(y_aug == c_idx))
            c_syn = c_total - c_real
            logger.info("  [%s] %s: Real=%d, Synthetic=%d (Total=%d)", s, c_name, c_real, c_syn, c_total)

        # Fit ONE joint t-SNE on real + synthetic
        emb_aug = run_tsne(Z_aug, seed=42)
        emb_real = emb_aug[:n_real]
        emb_syn = emb_aug[n_real:]
        y_syn = y_aug[n_real:]

        # Panel 1: Real points only
        ax_real = axes[row_idx, 0]
        for c_idx, c_name in enumerate(EMOTION_NAMES):
            mask = y_tr == c_idx
            ax_real.scatter(
                emb_real[mask, 0],
                emb_real[mask, 1],
                c=EMOTION_COLORS[c_idx],
                label=f"{c_name} (Real)",
                alpha=0.75,
                s=28,
                marker="o",
                edgecolors="none",
            )
        ax_real.set_title(
            f"{SETTING_LABELS[s]} - Real Training Latents Only (N={n_real})",
            fontsize=12,
            fontweight="bold",
        )
        ax_real.set_xlabel("t-SNE Dimension 1", fontsize=10)
        ax_real.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax_real.grid(True, linestyle="--", alpha=0.4)

        # Panel 2: Joint Real (circles) + Synthetic (triangles)
        ax_joint = axes[row_idx, 1]
        for c_idx, c_name in enumerate(EMOTION_NAMES):
            mask_r = y_tr == c_idx
            ax_joint.scatter(
                emb_real[mask_r, 0],
                emb_real[mask_r, 1],
                c=EMOTION_COLORS[c_idx],
                alpha=0.6,
                s=24,
                marker="o",
                edgecolors="none",
            )
            mask_s = y_syn == c_idx
            if np.any(mask_s):
                ax_joint.scatter(
                    emb_syn[mask_s, 0],
                    emb_syn[mask_s, 1],
                    c=EMOTION_COLORS[c_idx],
                    alpha=0.85,
                    s=28,
                    marker="^",
                    edgecolors="black",
                    linewidths=0.3,
                )
        ax_joint.set_title(
            f"{SETTING_LABELS[s]} - Joint Real (o) + RCS Synthetic (^) (N={len(Z_aug)})",
            fontsize=12,
            fontweight="bold",
        )
        ax_joint.set_xlabel("t-SNE Dimension 1", fontsize=10)
        ax_joint.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax_joint.grid(True, linestyle="--", alpha=0.4)

    # Create custom legend showing emotion colors + shapes
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"{name}", markerfacecolor=col, markersize=8)
        for name, col in zip(EMOTION_NAMES, EMOTION_COLORS.values())
    ] + [
        Line2D([0], [0], marker="o", color="w", label="Real Sample", markerfacecolor="gray", markersize=8),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="RCS Synthetic",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=8,
        frameon=True,
        fontsize=10,
        title="Emotion Class & Sample Type",
        title_fontsize=11,
    )
    fig.suptitle(
        "Figure 3: RCS Latent Subspace Synthesis under Moderate and Severe Imbalance (Train Split, Seed 42)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, "fig3_rcs_synthesis_tsne")


def plot_figure4_actor_check(y_test: np.ndarray, test_features: Dict[str, np.ndarray]):
    """
    Figure 4 (actor check): TEST latents (balanced), colored by actor (13 actors)
    next to the emotion-colored version (1x2 grid).
    """
    logger.info("Generating Figure 4: Actor Invariance Check...")
    latents_balanced = np.load(LATENTS_DIR / "balanced_fusion_v2_seed42.npz")["Z_te"]
    actors = test_features["actor"]
    unique_actors = np.unique(actors)

    emb = run_tsne(latents_balanced, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left Panel: Emotion-colored
    ax_emo = axes[0]
    for c_idx, c_name in enumerate(EMOTION_NAMES):
        mask = y_test == c_idx
        ax_emo.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=EMOTION_COLORS[c_idx],
            label=c_name,
            alpha=0.75,
            s=30,
            edgecolors="none",
        )
    ax_emo.set_title("Balanced Test Latents Colored by Emotion (6 Classes)", fontsize=12, fontweight="bold")
    ax_emo.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax_emo.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax_emo.grid(True, linestyle="--", alpha=0.4)
    ax_emo.legend(title="Emotion", frameon=True, fontsize=10, loc="upper right")

    # Right Panel: Actor-colored (13 actors)
    ax_act = axes[1]
    cmap = plt.colormaps["tab20"]
    for a_idx, act_id in enumerate(unique_actors):
        mask = actors == act_id
        ax_act.scatter(
            emb[mask, 0],
            emb[mask, 1],
            color=cmap(a_idx),
            label=f"Actor {act_id}",
            alpha=0.75,
            s=30,
            edgecolors="none",
        )
    ax_act.set_title(
        "Balanced Test Latents Colored by Actor Identity (13 Test Speakers)",
        fontsize=12,
        fontweight="bold",
    )
    ax_act.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax_act.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax_act.grid(True, linestyle="--", alpha=0.4)
    ax_act.legend(title="Actor ID", frameon=True, fontsize=8, ncol=2, loc="upper right")

    fig.suptitle(
        "Figure 4: Latent Clustering Verification — Emotion Content vs. Speaker Identity (Seed 42)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, "fig4_actor_check_tsne")


def plot_figure5_errors(y_test: np.ndarray):
    """
    Figure 5 (errors): lt_rho50 TEST latents, same embedding in 3 panels
    showing points misclassified by no_rebalance, balanced_softmax, and RCS-Optimal
    (correct = grey, wrong = red).
    """
    logger.info("Generating Figure 5: Classification Error Distribution (lt_rho50)...")
    latents_rho50 = np.load(LATENTS_DIR / "lt_rho50_fusion_v2_seed42.npz")["Z_te"]
    emb = run_tsne(latents_rho50, seed=42)

    with open(RESULTS_DIR / "test_predictions.json", "r", encoding="utf-8") as f:
        tp = json.load(f)

    methods = [
        ("no_rebalance", "Unweighted ERM (no_rebalance)"),
        ("balanced_softmax", "Balanced Softmax"),
        ("RCS-Optimal", "RCS-Optimal"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (m_key, m_title) in enumerate(methods):
        ax = axes[idx]
        preds = np.array(tp[f"lt_rho50/fusion_v2/{m_key}/seed_42"])
        correct = preds == y_test
        err_mask = ~correct
        n_err = int(np.sum(err_mask))
        acc = np.mean(correct) * 100

        # Plot correct first in subtle grey
        ax.scatter(
            emb[correct, 0],
            emb[correct, 1],
            c="#bdc3c7",
            label=f"Correct ({np.sum(correct)})",
            alpha=0.5,
            s=22,
            edgecolors="none",
        )
        # Plot errors in bright red
        ax.scatter(
            emb[err_mask, 0],
            emb[err_mask, 1],
            c="#e74c3c",
            label=f"Misclassified ({n_err})",
            alpha=0.85,
            s=32,
            edgecolors="black",
            linewidths=0.4,
        )

        ax.set_title(
            f"{m_title}\nAccuracy: {acc:.1f}% | Errors: {n_err} / {len(y_test)}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, fontsize=10, loc="upper right")

    fig.suptitle(
        r"Figure 5: Test Misclassification Patterns under Extreme Imbalance ($\rho=50$, Fusion V2, Seed 42)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    save_figure(fig, "fig5_errors_rho50_tsne")


def write_tsne_readme():
    """Writes results_cremad/tsne/README.md with methodological disclaimer."""
    readme_path = OUTPUT_DIR / "README.md"
    content = """# CREMA-D 2D Representation Visualizations (t-SNE)

## Methodological Disclaimer & Interpretation Guide

1. **Illustration Only**: These 2D t-SNE projections are generated strictly for qualitative visualization and exploratory illustration of high-dimensional feature and latent geometry.
2. **Distances & Density Caveats**: Non-linear manifold projections (t-SNE) do not preserve global pairwise Euclidean distances or relative cluster densities. Cluster sizes, cluster separations, and spatial distances across different regions are not directly interpretable as quantitative emotion proximity metrics.
3. **No Optimization on Visualizations**: No hyperparameters, architecture decisions, rebalancing strategies, or tuning procedures were guided, adjusted, or optimized based on these plots. All model selection was conducted strictly by validation Macro-F1 prior to visualization.
4. **Held-Out Test Display**: Test set embeddings are displayed strictly post-evaluation for qualitative inspection. No test data guided representation learning or sampling decisions.
5. **Deterministic Pipeline**: All panels utilize an identical, standardized decomposition pipeline:
   - Initial dimensionality reduction: `sklearn.decomposition.PCA(n_components=50, random_state=42)`
   - Manifold embedding: `sklearn.manifold.TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)`
   - Fixed emotion color mapping from `data/label_map.json`.

---

## Index of Figures

| Figure | Filename (PNG / PDF) | Split | Representation / Setting | Description |
| :--- | :--- | :---: | :--- | :--- |
| **Figure 1** | `fig1_settings_tsne` | Test | Fusion V2 across 4 Imbalance Settings | 2x2 grid comparing test latent geometry across balanced ($\rho \approx 1.2$), $\rho=10$, $\rho=20$, and $\rho=50$. |
| **Figure 2** | `fig2_modalities_tsne` | Test | Audio, Video, Concat, and Fused Latents | 4-panel comparison of modality geometry with original-space silhouette scores in panel titles. |
| **Figure 3** | `fig3_rcs_synthesis_tsne` | Train | Real vs. RCS Synthetic Latents ($\rho=10, 50$) | Joint t-SNE embedding comparing real training points (circles) to synthetic minority samples (triangles). |
| **Figure 4** | `fig4_actor_check_tsne` | Test | Balanced Fusion V2 Latents | Side-by-side comparison colored by emotion class (6 emotions) vs. actor identity (13 test speakers). |
| **Figure 5** | `fig5_errors_rho50_tsne` | Test | $\rho=50$ Test Predictions across Methods | Identical test embedding showing classification errors (red) vs. correct predictions (grey) for `no_rebalance`, `balanced_softmax`, and `RCS-Optimal`. |
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    logger.info("Saved %s", readme_path)


def main():
    logger.info("=" * 80)
    logger.info("Step 11: Generating CREMA-D t-SNE Figures and Artifacts")
    logger.info("=" * 80)

    # 1. Load test features
    test_feat = np.load(DATA_DIR / "features_cremad" / "test_features.npz")
    y_test = test_feat["y"]

    # 2. Generate Figures 1 through 5
    plot_figure1_settings(y_test)
    plot_figure2_modalities(y_test, test_feat)
    plot_figure3_rcs_synthesis()
    plot_figure4_actor_check(y_test, test_feat)
    plot_figure5_errors(y_test)

    # 3. Write README.md
    write_tsne_readme()

    logger.info("=" * 80)
    logger.info("SUCCESS: All 5 t-SNE figures (PNG + PDF) and README.md created in %s", OUTPUT_DIR)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
