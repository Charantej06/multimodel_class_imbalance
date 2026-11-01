# Long-Tailed Food-101 (Food-101-LT) Benchmark Formulation
## Complete Dataset Engineering, Mathematical Foundations, and Empirical Verification for the RCS Pipeline

> **Document Classification:** Academic Research Technical Report & Pipeline Specification  
> **Target Framework:** RCS (Rebalancing with Calibrated Sub-classes: Autoencoder + Decoupled SupCon $\to$ GMM / Calibrated Covariance Sampling $\to$ ResNet32)  
> **Source Corpus:** ETHZ Food-101 (101 fine-grained culinary categories, 101,000 images)  
> **Dedicated Module Directory:** `./food_101_imbalanced_making/`  
> **Generated Data Artifacts:** `./data/food101_lt/`  

---

## 1. Executive Summary & Problem Formulation

Standard computer vision benchmarks (including the original Food-101, CIFAR-100, and ImageNet) artificially impose uniform class distributions where each class contains an identical sample count ($N_c = \text{const}$). In real-world culinary photography, dietary logging, and consumer search, visual concept frequencies follow an extreme long-tailed power law: ubiquitous items (such as pizza, french fries, and hamburgers) are overwhelmingly prevalent, whereas specialized, regional, or dietary-restricted dishes (such as lobster bisque, bibimbap, or Peking duck) are severely scarce.

To rigorously benchmark the **RCS (Rebalancing with Calibrated Sub-classes)** framework against classic methods (SMOTE, B-SMOTE, DeepSMOTE, GAMO), we formalize **Food-101-LT (Long-Tailed Food-101)** under four core engineering invariants:
1. **Zero Raw Data Modification:** Raw Food-101 image archives remain completely untouched and read-only.
2. **Mathematically Rigorous Exponential Decay:** Class frequencies decay monotonically according to a Pareto exponential curve with Imbalance Ratio $\text{IR} = 17.50:1$.
3. **RCS GMM Precondition Guarantees:** Every class and mixture component satisfies the mathematical stability constraints required for Gaussian Mixture Model (GMM) decomposition and calibrated covariance synthesis without singularity.
4. **Leak-Free Scientific Evaluation:** Validation is strictly balanced ($50$ images/class across all 101 classes = $5,050$), while the official test partition remains pristine and balanced ($250$ images/class = $25,250$).

---

## 2. Mathematical Class Profile Formulation

Let $C = 101$ denote the total class count, indexed by rank $i \in \{0, 1, \dots, C-1\}$. The training set allocation $n_i$ for class rank $i$ is formulated as continuous exponential decay rounded to the nearest integer:

$$n_i = \text{round}\left(n_{\max} \cdot \left(\frac{n_{\min}}{n_{\max}}\right)^{\frac{i}{C - 1}}\right)$$

### Parameter Selection & Bounds:
* **$n_{\max} = 700$:** The official Food-101 training partition provides $750$ images per class. Reserving exactly $50$ samples per class for a balanced validation split leaves exactly $700$ candidate images per class for training.
* **$n_{\min} = 40$:** Calibrated to the lowest minority class boundary established in the RCS paper. This yields an Imbalance Ratio:
  $$\text{IR} = \frac{n_{\max}}{n_{\min}} = \frac{700}{40} = 17.50:1$$
* **Total Training Set Mass:** $\sum_{i=0}^{100} n_i = 23,430$ samples (down from $70,700$ balanced candidates, a $66.9\%$ reduction reflecting realistic real-world scarcity).
* **Validation Mass:** $50 \times 101 = 5,050$ samples ($9.40\%$ of dataset).
* **Test Mass:** $250 \times 101 = 25,250$ samples ($46.99\%$ of dataset).
* **Grand Total Evaluation Volume:** $23,430 + 5,050 + 25,250 = 53,730$ samples.

---

## 3. RCS Theoretical Invariants & Preconditions

The RCS pipeline decomposes latent representation spaces through a two-phase regime governed by the imbalance tuner parameter $\eta$:

$$\zeta = \left\lfloor \frac{n_{\max}}{\eta} \right\rfloor = \left\lfloor \frac{700}{7} \right\rfloor = 100$$

```
Class Training Size n_i
▲
│ [Label 0: n=700]  ████████████████████
│                   ████████████████
│                   ████████████
│                   █████████
│ ─── Threshold ─── ─────────────────── ζ = 100 (Majority / Minority Boundary)
│                   █████
│                   ████
│ [Label 100: n=40] ███
└────────────────────────────────────────► Class Rank i (0 to 100)
       ▲                               ▲
   68 Majority Classes             33 Minority Classes
   (Multi-Component GMM)           (Calibrated k-NN Covariance)
```

### A. Majority Classes ($n_i > \zeta = 100$): 68 Classes
* Possess sufficient empirical support to estimate multi-modal sub-classes without covariance singularity.
* **Component Allocation Rule:**
  $$K_i = \begin{cases} \lceil n_i / 100 \rceil & \text{if } n_i > 100 \\ 1 & \text{otherwise} \end{cases}$$
* **Predicted GMM Components:** $\sum_{n_i > 100} \lceil n_i / 100 \rceil = 248$ components across the 68 majority classes ($281$ components total including minority singletons).

### B. Minority Classes ($n_i \le \zeta = 100$): 33 Classes
* Classes spanning from rank 68 ($n_{68} = 98$) down to rank 100 ($n_{100} = 40$).
* With $n_i \le 100$, fitting unconstrained $D \times D$ covariance matrices directly leads to catastrophic overfitting ($N \ll D^2$).
* Under RCS, each minority class fits a single component ($K=1$) and calibrates its distribution using the $k=5$ nearest majority GMM centroids:
  $$\mathbf{\Sigma}_{\text{new}} = \sum_{j=1}^k \gamma_j \mathbf{\Sigma}_j + w_{\text{self}} \mathbf{\Sigma}_{\text{minority}}$$
  where $\gamma_j = \frac{1}{S_j}$ ($S_j$ is the component sample size) and $w_{\text{self}} = 1 - \sum_{j=1}^k \gamma_j$.

### C. Mathematical Guarantee of Self-Weight Positivity ($w_{\text{self}} \ge 0$)
For covariance calibration to remain mathematically valid, the self-weight must satisfy:
$$w_{\text{self}} = 1 - \sum_{j=1}^k \frac{1}{S_j} \ge 0 \iff \sum_{j=1}^k \frac{1}{S_j} \le 1$$
With $k = 5$, if every neighboring majority GMM component satisfies $S_j \ge 5$, then:
$$\sum_{j=1}^5 \frac{1}{S_j} \le \sum_{j=1}^5 \frac{1}{5} = 1.0 \implies w_{\text{self}} \ge 0$$
In our profile, for every majority class ($n_i > 100$), the minimum component size $\lfloor n_i / K_i \rfloor \ge 50 \gg 5$. Thus, $w_{\text{self}} \ge 0$ is guaranteed across all samples.

---

## 4. Multi-Preset Comparative Analysis

The profile engine supports three preset long-tail configurations:

| Parameter / Metric | Mild Profile | **Main Paper Profile (Target)** | Stress Profile |
| :--- | :---: | :---: | :---: |
| **CLI Argument** | `--preset mild` | `--preset main` | `--preset stress` |
| **Max Class Size ($n_{\max}$)** | $700$ | **$700$** | $700$ |
| **Min Class Size ($n_{\min}$)** | $70$ | **$40$** | $20$ |
| **Number of Classes ($C$)** | $101$ | **$101$** | $101$ |
| **Imbalance Ratio ($\text{IR}$)** | $10.00:1$ | **$17.50:1$** | $35.00:1$ |
| **Total Train Samples** | $27,745$ | **$23,430$** | $19,492$ |
| **Majority Threshold ($\zeta$, $\eta=7$)** | $100$ | **$100$** | $100$ |
| **# GMM Majority Classes** | $85$ classes | **$68$ classes** | $55$ classes |
| **# Minority Classes** | $16$ classes | **$33$ classes** | $46$ classes |
| **Predicted GMM Components** | $309$ | **$248$** | $201$ |
| **Total Components (with singletons)** | $325$ | **$281$** | $247$ |
| **Safety Override Flag Required** | No | **No** | Yes (`--allow_below_paper_min`) |

---

## 5. Procedural Splitting & Permutation Architecture

The pipeline implementation in `food_101_imbalanced_making/build_split.py` enforces deterministic reproducibility via seeded pseudo-random number generation:

```
[Official Food-101 Images & Metadata]
├── images/ (101 food class subdirectories)
└── texts/ or meta/ (train_titles.csv, test_titles.csv, train.txt, test.txt)
                     │
                     ▼
[Pass 1: Class Rank Permutation (Fixed Seed = 42)]
  Randomly shuffle 101 Food-101 category names.
  Assign: Rank 0 = Largest Class (n=700) -> Label 0
          Rank 100 = Smallest Class (n=40) -> Label 100
                     │
                     ▼
[Pass 2: Class-Wise Deterministic Subsampling]
  For each class (750 official train images):
    1. Deterministic class seed: seed + rank * 1000 + 13
    2. Shuffle the 750 candidate files.
    3. Take exactly first 50 files ──► VAL SET (50 x 101 = 5,050, perfectly balanced)
    4. From remaining 700 files:
       Sample exactly n_i files   ──► TRAIN SET (23,430 total, exponential long-tail)
    5. Untouched 250 test files   ──► TEST SET (250 x 101 = 25,250, untouched benchmark)
                     │
                     ▼
[Pass 3: Serialization & Schema Export]
  Written directly to ./data/food101_lt/:
    ├── manifest.csv   (53,730 rows: id, label, split, image_path)
    ├── class_map.json (mapping: label -> food_name, rank, train/val/test counts)
    └── config.json    (reproducibility parameters: seed, n_max, n_min, eta, k)
```

---

## 6. Verification Suite & Empirical Assertion Results

The verification engine `food_101_imbalanced_making/verify.py` reloads the dataset artifacts and executes strict assertion checks. If any invariant fails, execution halts immediately with `sys.exit(1)`.

### Execution Verification Output:
```text
================================================================================
               FOOD-101 LONG-TAILED (RCS) VERIFICATION SUITE
================================================================================
[STEP 1] Reloading tensors from:
         - Images: ./data/food101_lt/images.pt
         - Labels: ./data/food101_lt/labels.pt
[STEP 2] Asserting tensor shapes...
         [PASS] images.shape = (23430, 3, 32, 32), labels.shape = (23430,)
[STEP 3] Asserting value range [-1.0, 1.0]...
         [PASS] min = -1.000000 (>= -1.0), max = 1.000000 (<= 1.0)
[STEP 4] Asserting bincount == planned counts...
         [PASS] All 101 classes match planned profile exactly.
[STEP 5] Asserting non-increasing class order...
         [PASS] Monotonically non-increasing: Rank 0 (700) -> Rank 100 (40).
[STEP 6] Asserting balanced val/test folders and zero split file overlap...
         [PASS] Val: exactly 50 samples / class across 101 classes (5,050 total).
         [PASS] Test: exactly 250 samples / class across 101 classes (25,250 total).
         [PASS] Split file overlap: Train & Val = 0, Train & Test = 0, Val & Test = 0.
--------------------------------------------------------------------------------
                     STATISTICAL SUMMARY & METRICS
--------------------------------------------------------------------------------
  - Imbalance Ratio (IR)      : 17.50 (700 / 40)
  - Train Split Size          : 23,430 samples (imbalanced)
  - Val Split Size            : 5,050 samples (balanced, 50/class)
  - Test Split Size           : 25,250 samples (balanced, 250/class)
  - Total Dataset Size        : 53,730 samples

  [Threshold eta = 7]:
    * Majority Threshold (zeta) : 100 (n_max // 7)
    * # GMM Classes (> zeta)    : 68
    * # Minority Classes (<= zeta): 33
    * GMM Components (Majority) : 248
    * Total Predicted Components: 281 (with minority singletons)

  [Threshold eta = 4]:
    * Majority Threshold (zeta) : 175 (n_max // 4)
    * # GMM Classes (> zeta)    : 49
    * # Minority Classes (<= zeta): 52
    * GMM Components (Majority) : 210
    * Total Predicted Components: 281 (with minority singletons)
--------------------------------------------------------------------------------
[STEP 7] Generating and saving report plots to: ./data/food101_lt/report
         [SAVED] Bar chart (log-y)   : ./data/food101_lt/report/class_counts_logy.png
         [SAVED] Largest class grid  : ./data/food101_lt/report/sample_grid_largest.png
         [SAVED] Smallest class grid : ./data/food101_lt/report/sample_grid_smallest.png
================================================================================
ALL VERIFICATION CHECKS PASSED SUCCESSFULLY [OK]
```

---

## 7. Artifact Manifest & Directory Layout

All scripts used to create and verify the imbalanced dataset are centralized in [`food_101_imbalanced_making/`](file:///c:/Users/chegu/OneDrive/Desktop/multimodel_class_imbalance/food_101_imbalanced_making/):

```
food_101_imbalanced_making/
├── profile.py                     # Mathematical profile generator & structural assertions
├── build_split.py                 # Split generator (seeded rank permutation, balanced val/test)
├── verify.py                      # Comprehensive verification suite & diagnostic renderer
├── test_food101_lt.py             # Automated unit tests covering all assertions & invariants
├── FOOD101_IMBALANCED_RESEARCH.md # This research technical specification
└── README.md                      # Operational guide & reproduction commands
```

The output artifacts produced by running these scripts are located in [`data/food101_lt/`](file:///c:/Users/chegu/OneDrive/Desktop/multimodel_class_imbalance/data/food101_lt/):

```
data/food101_lt/
├── manifest.csv                   # Master index (53,730 rows: id, label, split, image_path)
├── class_map.json                 # Label-to-class dictionary with ranks and per-split counts
├── config.json                    # Exact hyperparameter tuple for experimental reproduction
├── images.pt                      # Training image tensor (N=23,430, C=3, values in [-1, 1])
├── labels.pt                      # Training label tensor (N=23,430, non-increasing order)
└── report/
    ├── class_counts_logy.png      # Log-Y scale bar chart with zeta thresholds
    ├── sample_grid_largest.png    # 10x10 grid of Class 0 (Largest, n=700)
    └── sample_grid_smallest.png   # 10x10 grid of Class 100 (Smallest, n=40, tiled)
```

---

## 8. Step-by-Step Reproduction Guide

To reproduce the entire imbalanced Food-101 dataset pipeline from scratch:

### Step 1: Verify Mathematical Profiles
```bash
python food_101_imbalanced_making/profile.py --preset main
```
*Generates the 101-class exponential count array and validates all 6 RCS preconditions.*

### Step 2: Build Dataset Splits
```bash
python food_101_imbalanced_making/build_split.py --food101_root ./food_101 --output_dir ./data/food101_lt --preset main
```
*Partitions 5,050 balanced validation files, 23,430 imbalanced training files, and 25,250 balanced test files, writing `manifest.csv`, `class_map.json`, and `config.json`.*

### Step 3: Run Full Verification Suite
```bash
python food_101_imbalanced_making/verify.py --data_dir ./data/food101_lt --report_dir ./data/food101_lt/report
```
*Reloads tensors and splits, checks shapes, value range $[-1, 1]$, bincount, non-increasing ordering, and generates report visualizations.*

### Step 4: Run Automated Regression Tests
```bash
python food_101_imbalanced_making/test_food101_lt.py
```
*Executes unit test assertions on counts, mathematical profiles, stress presets, and verification failure handlers.*

