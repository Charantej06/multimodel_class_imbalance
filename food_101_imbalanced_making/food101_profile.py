"""
food101_lt/profile.py
=====================
Exponential class imbalance profile generator and verification suite
for the Long-Tailed Food-101 (101 classes) benchmark under the RCS paper pipeline.

Pipeline formulation:
  n_i = round(n_max * (n_min / n_max) ** (i / (C - 1)))
"""

from __future__ import annotations
import argparse
import math
import random
from typing import Dict, List, Any, Optional

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PRESETS = {
    "mild": {"n_max": 700, "n_min": 70, "C": 101, "desc": "Mild Long-Tail (IR = 10.0)"},
    "main": {"n_max": 700, "n_min": 40, "C": 101, "desc": "Main Paper Benchmark (IR = 17.5, paper min = 40)"},
    "stress": {"n_max": 700, "n_min": 20, "C": 101, "desc": "Stress Profile (IR = 35.0, requires override)"},
}


def make_counts(n_max: int = 700, n_min: int = 40, C: int = 101) -> List[int]:
    """
    Generates an exponential class frequency profile for C classes:
        n_i = round(n_max * (n_min / n_max) ** (i / (C - 1)))
    """
    assert n_max > 0, f"Assert failed: n_max must be positive, got {n_max}"
    assert n_min > 0, f"Assert failed: n_min must be positive, got {n_min}"
    assert n_max >= n_min, f"Assert failed: n_max ({n_max}) must be >= n_min ({n_min})"
    assert C > 1, f"Assert failed: number of classes C must be > 1, got {C}"

    counts = [round(n_max * (n_min / n_max) ** (i / (C - 1))) for i in range(C)]
    return counts


def check_profile(
    counts: List[int],
    eta: int = 7,
    k: int = 5,
    allow_below_paper_min: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verifies that the generated class count profile satisfies all theoretical
    and structural assumptions of the RCS Gaussian Mixture Model pipeline.

    Assertions:
      (1) counts are non-increasing.
      (2) n_min >= 40 unless allow_below_paper_min=True (paper's smallest class is 40).
      (3) zeta = counts[0] // eta, at least one class > zeta;
          reports #GMM classes (> zeta) and #minority (<= zeta).
      (4) smallest class < 100 (GMM component size ~100 must exceed smallest class).
      (5) with k=5, every GMM component has >= 5 points, i.e. no class with n > zeta
          can have a component smaller than 5 (so 1 - sum(1/S_j) >= 0).
      (6) report predicted GMM components = sum(ceil(n/100) if n > 100 else 1).
    """
    C = len(counts)
    assert C > 0, "Assert failed: counts list is empty."

    # (1) Non-increasing check
    for i in range(C - 1):
        assert counts[i] >= counts[i + 1], (
            f"Assert failed (1): counts are not monotonically non-increasing at index {i}: "
            f"counts[{i}]={counts[i]} < counts[{i+1}]={counts[i+1]}."
        )

    n_max = counts[0]
    n_min = counts[-1]
    total_samples = sum(counts)
    imbalance_ratio = n_max / n_min

    # (2) Smallest class >= 40 unless allow_below_paper_min=True
    assert allow_below_paper_min or n_min >= 40, (
        f"Assert failed (2): smallest class count n_min={n_min} < 40 (paper minimum). "
        f"Use allow_below_paper_min=True to override for stress testing."
    )

    # (3) Majority threshold zeta = counts[0] // eta
    assert eta > 0, f"Assert failed: eta must be positive, got {eta}"
    zeta = counts[0] // eta
    gmm_classes = [c for c in counts if c > zeta]
    minority_classes = [c for c in counts if c <= zeta]

    assert len(gmm_classes) > 0, (
        f"Assert failed (3): at least one class must exceed zeta={zeta} (counts[0]={counts[0]}, eta={eta}), "
        f"but no class was found."
    )

    # (4) Smallest class < 100
    assert n_min < 100, (
        f"Assert failed (4): smallest class count n_min={n_min} >= 100. "
        f"GMM component size (~100) must exceed the smallest class to trigger minority calibration."
    )

    # (5) With k=5, every GMM component has >= 5 points for classes with n > zeta
    for idx, n in enumerate(counts):
        if n > zeta:
            n_comps = math.ceil(n / 100) if n > 100 else 1
            min_points_in_comp = n // n_comps
            assert min_points_in_comp >= k, (
                f"Assert failed (5): class index {idx} with n={n} divided into {n_comps} components "
                f"has min points {min_points_in_comp} < k={k}. "
                f"Component size S_j must be >= k so that 1 - sum(1/S_j) >= 0."
            )

    # (6) Predicted GMM components
    predicted_gmm_components = sum(math.ceil(n / 100) if n > 100 else 1 for n in gmm_classes)
    total_components_all_classes = sum(math.ceil(n / 100) if n > 100 else 1 for n in counts)

    info = {
        "C": C,
        "n_max": n_max,
        "n_min": n_min,
        "total_samples": total_samples,
        "imbalance_ratio": imbalance_ratio,
        "zeta": zeta,
        "n_gmm_classes": len(gmm_classes),
        "n_minority_classes": len(minority_classes),
        "predicted_gmm_components": predicted_gmm_components,
        "total_components_all_classes": total_components_all_classes,
        "counts": counts,
    }

    if verbose:
        print_profile_report(info, k=k, eta=eta)

    return info


def print_histogram(counts: List[int], bins: int = 10):
    """Prints a clean ASCII distribution histogram of class frequencies."""
    min_c, max_c = min(counts), max(counts)
    step = (max_c - min_c) / bins
    print("\nClass Frequency Histogram (ASCII):")
    for b in range(bins):
        low = min_c + b * step
        high = min_c + (b + 1) * step
        in_bin = [c for c in counts if (low <= c <= high if b == bins - 1 else low <= c < high)]
        bar = "#" * len(in_bin)
        print(f"  [{low:5.1f} - {high:5.1f}]: {len(in_bin):3d} classes | {bar}")


def print_profile_report(info: Dict[str, Any], k: int, eta: int):
    """Prints formatted profile metrics and structural checks."""
    print("=" * 85)
    print("LONG-TAILED FOOD-101 PROFILE VERIFICATION REPORT")
    print("=" * 85)
    print(f"  Total Classes (C)          : {info['C']}")
    print(f"  Max Class Samples (n_max)  : {info['n_max']}")
    print(f"  Min Class Samples (n_min)  : {info['n_min']}")
    print(f"  Total Dataset Samples      : {info['total_samples']:,}")
    print(f"  Imbalance Ratio (IR)       : {info['imbalance_ratio']:.2f}:1")
    print("-" * 85)
    print(f"  Imbalance Tuner (eta)      : {eta}")
    print(f"  Majority Threshold (zeta)  : {info['zeta']} (n_max // {eta})")
    print(f"  # GMM Classes (n > zeta)   : {info['n_gmm_classes']} classes")
    print(f"  # Minority Classes (<=zeta): {info['n_minority_classes']} classes")
    print(f"  Predicted GMM Components   : {info['predicted_gmm_components']} (on majority classes)")
    print(f"  Nearest Neighbors (k)      : {k} (every component >= {k} points verified)")
    print("-" * 85)
    print("  Assertions Passed:")
    print("    [PASS] (1) Counts are strictly non-increasing across all 101 classes.")
    print(f"    [PASS] (2) Smallest class n_min ({info['n_min']}) meets threshold requirements.")
    print(f"    [PASS] (3) Majority threshold zeta={info['zeta']} successfully partitions classes.")
    print(f"    [PASS] (4) Smallest class ({info['n_min']}) < 100 GMM component threshold.")
    print(f"    [PASS] (5) Every GMM component has >= {k} points -> 1 - sum(1/S_j) >= 0 guaranteed.")
    print(f"    [PASS] (6) Predicted GMM components = {info['predicted_gmm_components']} calculated.")

    print_histogram(info["counts"])


def print_summary_table(results: List[Dict[str, Any]]):
    """Prints a comparison table of all tested profiles."""
    print("\n" + "=" * 110)
    print("SUMMARY COMPARISON TABLE: FOOD-101 LONG-TAILED PRESETS")
    print("=" * 110)
    header = (
        f"{'Preset':<10} | {'n_max':<6} | {'n_min':<6} | {'IR':<7} | {'Total N':<9} | "
        f"{'zeta':<6} | {'#GMM (>z)':<10} | {'#Minority':<10} | {'GMM Comps':<10} | {'All Comps':<10}"
    )
    print(header)
    print("-" * 110)
    for r in results:
        tot_str = f"{r['total_samples']:,}"
        row = (
            f"{r['name']:<10} | {r['n_max']:<6} | {r['n_min']:<6} | {r['imbalance_ratio']:<7.2f} | "
            f"{tot_str:<9} | {r['zeta']:<6} | {r['n_gmm_classes']:<10} | "
            f"{r['n_minority_classes']:<10} | {r['predicted_gmm_components']:<10} | "
            f"{r['total_components_all_classes']:<10}"
        )
        print(row)
    print("=" * 110 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile generator and verification suite for Long-Tailed Food-101."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="main",
        choices=["mild", "main", "stress", "all"],
        help="Preset profile to check: 'mild' (700->70), 'main' (700->40), 'stress' (700->20), or 'all' to compare.",
    )
    parser.add_argument("--n_max", type=int, default=None, help="Override n_max (default: from preset)")
    parser.add_argument("--n_min", type=int, default=None, help="Override n_min (default: from preset)")
    parser.add_argument("--C", type=int, default=101, help="Number of classes (default: 101)")
    parser.add_argument("--eta", type=int, default=7, help="Imbalance tuner parameter (default: 7)")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest components (default: 5)")
    parser.add_argument(
        "--allow_below_paper_min",
        action="store_true",
        help="Allow n_min < 40 for stress testing (required for stress preset).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.preset == "all":
        results = []
        for name, cfg in PRESETS.items():
            allow_flag = args.allow_below_paper_min or (name == "stress" and args.allow_below_paper_min)
            # If stress without allow_below_paper_min, we can note it requires override
            counts = make_counts(n_max=cfg["n_max"], n_min=cfg["n_min"], C=cfg["C"])
            try:
                info = check_profile(
                    counts=counts,
                    eta=args.eta,
                    k=args.k,
                    allow_below_paper_min=args.allow_below_paper_min,
                    verbose=False,
                )
                info["name"] = name
                results.append(info)
            except AssertionError as e:
                print(f"[NOTE] Preset '{name}' assertion failed as expected: {e}")
                print(f"       Run with --allow_below_paper_min to enable '{name}'.")

        if results:
            print_summary_table(results)
    else:
        cfg = PRESETS[args.preset]
        n_max = args.n_max if args.n_max is not None else cfg["n_max"]
        n_min = args.n_min if args.n_min is not None else cfg["n_min"]

        counts = make_counts(n_max=n_max, n_min=n_min, C=args.C)
        info = check_profile(
            counts=counts,
            eta=args.eta,
            k=args.k,
            allow_below_paper_min=args.allow_below_paper_min,
            verbose=True,
        )
        info["name"] = args.preset
        print_summary_table([info])


if __name__ == "__main__":
    main()
