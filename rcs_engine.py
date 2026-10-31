"""
rcs_engine.py
=============
Modality- and class-count-agnostic Latent RCS Rebalancing Engine for Multimodal Emotion Recognition.

Formulation (RCS-Optimal):
  1. Subspace PCA Projection (D -> pca_dim) to eliminate the curse of dimensionality.
     Clamps pca_dim to min(pca_dim, smallest_train_class / 4) with a logged warning if clamped.
  2. Pass 1: GMM Sub-class Partitioning on majority classes (n_comps = ceil(N/100))
     with regularized diagonal covariance (reg_covar=1e-3) preventing matrix singularity.
  3. Pass 2: Calibrated k-NN (k=5) Gaussian Covariance Generation on minority classes
     using inverse-class weights to anchor synthesis along true decision frontiers.
  4. Target Capping: Caps synthetic share at 85% of any class (n_syn / (n_real + n_syn) <= 0.85).
  5. Subspace Inversion: Synthetic vectors mapped back to full D space via PCA.
"""

from __future__ import annotations
import math
import logging
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import scipy.spatial as sp
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

logger = logging.getLogger("rcs_engine")


def ensure_psd(cov: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Guards against NaN/inf and non-positive semi-definite covariances."""
    if np.isnan(cov).any() or np.isinf(cov).any():
        cov = np.nan_to_num(cov, nan=eps, posinf=1.0, neginf=-1.0)
    cov = (cov + cov.T) / 2.0
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        if np.any(eigvals < eps):
            eigvals = np.maximum(eigvals, eps)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    except np.linalg.LinAlgError:
        cov = cov + np.eye(cov.shape[0]) * eps
    return (cov + cov.T) / 2.0


def knn(X: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """Finds top-k nearest component indices based on Euclidean distance."""
    k_val = min(k, len(X))
    tree = sp.KDTree(X)
    _, ind = tree.query(x.reshape(1, -1), k=k_val)
    return ind[0]


def apply_rcs_optimal(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    label_names: Optional[List[str]] = None,
    pca_dim: int = 64,
    covariance_type: str = "diag",
    target_cap: int = 2500,
    eta: Optional[int] = None,
    k: int = 5,
    seed: int = 42,
    max_synthetic_share: Optional[float] = 0.85,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Applies the winning RCS-Optimal configuration to input latent vectors.

    Parameters
    ----------
    Z_train : (N, D) array of real latent representations (e.g. 512-d).
    y_train : (N,) array of integer emotion class labels.
    label_names : Optional list of class names (length equal to num_classes).
    pca_dim : Latent subspace dimension (default 64). Clamped to min(pca_dim, smallest_train_class / 4).
    covariance_type : 'diag' (recommended) or 'full'.
    target_cap : Maximum synthetic sample target per class (default 2500).
    eta : Imbalance tuner (threshold = target // eta). Defaults to num_classes if None.
    k : Nearest GMM components to calibrate with (default 5).
    seed : Random state seed.
    max_synthetic_share : Maximum allowed synthetic share in any class (default 0.85 = 85%).

    Returns
    -------
    Z_balanced : (N_balanced, D) Stacked real + synthetic latent features.
    y_balanced : (N_balanced,) Integer labels for the balanced dataset.
    info : Diagnostic dictionary with sample counts and generation statistics.
    """
    np.random.seed(seed)
    D = Z_train.shape[1]
    counts = np.bincount(y_train)
    num_classes = len(counts)

    if label_names is None:
        label_names = [f"class_{i}" for i in range(num_classes)]

    smallest_train_class = int(counts.min())

    # Clamp pca_dim to min(pca_dim, smallest_train_class / 4) with a logged warning
    if pca_dim is not None:
        max_allowed_pca = max(1, int(smallest_train_class / 4))
        if pca_dim > max_allowed_pca:
            logger.warning(
                "Clamping pca_dim from %d to %d (smallest class has %d samples; max allowed is smallest_class / 4)",
                pca_dim, max_allowed_pca, smallest_train_class,
            )
            pca_dim = max_allowed_pca

    # Dynamic eta default
    if eta is None:
        eta = num_classes

    target = min(int(counts.max()), target_cap) if target_cap is not None else int(counts.max())
    threshold = target // eta

    counts_before = {label_names[i]: int(counts[i]) for i in range(num_classes)}

    # 1. PCA Subspace Projection
    pca_model = None
    if pca_dim is not None and pca_dim < D:
        pca_model = PCA(n_components=pca_dim, random_state=seed)
        Z_work = pca_model.fit_transform(Z_train)
        D_work = pca_dim
    else:
        Z_work = Z_train.copy()
        D_work = D

    after_gauss_X = []
    after_gauss_Y = []
    mean_gauss = []
    covariance_gauss = []
    class_samples = []

    # 2. PASS 1: GMM Fitting on Majority Classes (count > threshold)
    for i in range(num_classes):
        xclass = Z_work[y_train == i]
        n_real = len(xclass)
        c_name = label_names[i]

        if n_real > threshold:
            n_comps = math.ceil(n_real / 100) if n_real > 100 else 1
            gm = GaussianMixture(n_components=n_comps, random_state=seed, covariance_type=covariance_type, reg_covar=1e-3)
            gm.fit(xclass)

            n_need = max(0, target - n_real)
            # Cap synthetic share at max_synthetic_share (85%)
            if max_synthetic_share is not None and 0.0 < max_synthetic_share < 1.0:
                max_syn = int(math.floor(n_real * max_synthetic_share / (1.0 - max_synthetic_share)))
                if n_need > max_syn:
                    logger.info("Capping synthetic samples for majority class '%s' from %d to %d (max share %.1f%%)", c_name, n_need, max_syn, max_synthetic_share * 100)
                    n_need = max_syn

            if n_need > 0:
                samples_per_comp = math.ceil(n_need / n_comps)
                gen_list = []
                for comp_idx in range(n_comps):
                    c_mean = gm.means_[comp_idx]
                    if covariance_type == "full":
                        c_cov = ensure_psd(gm.covariances_[comp_idx])
                    else:
                        c_cov = np.diag(np.maximum(gm.covariances_[comp_idx], 1e-4))
                    samples = np.random.multivariate_normal(c_mean, c_cov, samples_per_comp)
                    gen_list.append(samples)
                all_s = np.vstack(gen_list)[:n_need]
                after_gauss_X.append(all_s)
                after_gauss_Y.append(np.full(n_need, i, dtype=np.int64))

            mean_gauss.append(gm.means_)
            if covariance_type == "full":
                cov_list = [ensure_psd(gm.covariances_[c_idx]) for c_idx in range(n_comps)]
            else:
                cov_list = [np.diag(np.maximum(gm.covariances_[c_idx], 1e-4)) for c_idx in range(n_comps)]
            covariance_gauss.append(np.stack(cov_list, axis=0))
            class_samples.extend([n_real] * n_comps)

    if len(mean_gauss) == 0:
        # Fallback if no classes exceeded threshold
        mean_gauss = np.zeros((1, D_work))
        covariance_gauss = np.zeros((1, D_work, D_work))
        class_samples = np.array([1.0])
    else:
        mean_gauss = np.vstack(mean_gauss)
        covariance_gauss = np.vstack(covariance_gauss)
        class_samples = np.array(class_samples, dtype=np.float64)

    inv_weights = 1.0 / np.maximum(class_samples, 1.0)

    # 3. PASS 2: Calibrated k-NN Gaussian Generation on Minority Classes (count <= threshold)
    for i in range(num_classes):
        xclass = Z_work[y_train == i]
        n_real = len(xclass)
        c_name = label_names[i]

        if n_real <= threshold and n_real > 0:
            n_need = max(0, target - n_real)
            # Cap synthetic share at max_synthetic_share (85%)
            if max_synthetic_share is not None and 0.0 < max_synthetic_share < 1.0:
                max_syn = int(math.floor(n_real * max_synthetic_share / (1.0 - max_synthetic_share)))
                if n_need > max_syn:
                    logger.info("Capping synthetic samples for minority class '%s' from %d to %d (max share %.1f%%)", c_name, n_need, max_syn, max_synthetic_share * 100)
                    n_need = max_syn

            if n_need <= 0:
                continue

            gen_elt_per_elt = math.ceil(n_need / n_real)
            gm_minority = GaussianMixture(n_components=1, random_state=seed, covariance_type=covariance_type, reg_covar=1e-3)
            gm_minority.fit(xclass)
            if covariance_type == "full":
                class_cov = ensure_psd(gm_minority.covariances_[0])
            else:
                class_cov = np.diag(np.maximum(gm_minority.covariances_[0], 1e-4))

            gen_X_class = []
            for x in xclass:
                idx = knn(mean_gauss, x, k)
                w_components = inv_weights[idx]
                sum_weights = float(np.sum(w_components))
                w_self = max(0.0, 1.0 - sum_weights)

                # Calibrated Mean
                new_mean = np.dot(w_components, mean_gauss[idx]) + w_self * x

                # Calibrated Covariance
                temp_cov = [w_components[iter_idx] * covariance_gauss[idx[iter_idx]] for iter_idx in range(len(idx))]
                sum_cov = np.sum(np.array(temp_cov), axis=0)
                new_cov = sum_cov + w_self * class_cov
                if covariance_type == "full":
                    new_cov = ensure_psd(new_cov)
                else:
                    new_cov = np.diag(np.maximum(np.diag(new_cov), 1e-4))

                gen_elts = np.random.multivariate_normal(new_mean, new_cov, gen_elt_per_elt)
                gen_X_class.append(gen_elts)

            gen_X_class = np.vstack(gen_X_class)[:n_need]
            after_gauss_X.append(gen_X_class)
            after_gauss_Y.append(np.full(n_need, i, dtype=np.int64))

    if len(after_gauss_X) > 0:
        Z_syn_work = np.vstack(after_gauss_X)
        y_syn = np.hstack(after_gauss_Y)
    else:
        Z_syn_work = np.zeros((0, D_work))
        y_syn = np.zeros(0, dtype=np.int64)

    # 4. Invert back to full latent space
    if pca_model is not None and len(Z_syn_work) > 0:
        Z_syn = pca_model.inverse_transform(Z_syn_work)
    else:
        Z_syn = Z_syn_work

    Z_balanced = np.concatenate([Z_train, Z_syn], axis=0)
    y_balanced = np.concatenate([y_train, y_syn], axis=0)

    counts_after = {label_names[i]: int(np.sum(y_balanced == i)) for i in range(num_classes)}

    info = {
        "target": target,
        "pca_dim": pca_dim,
        "covariance_type": covariance_type,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "n_synthetic_total": len(Z_syn),
        "n_balanced_total": len(Z_balanced),
        "max_synthetic_share": max_synthetic_share,
    }

    return Z_balanced, y_balanced, info
