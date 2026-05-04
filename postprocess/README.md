# Post-Processing Module

This module provides tools for correcting biased annotation distributions using various post-processing methods, including the CleverLabel algorithm.

## Usage

### Detailed Experiment (`run_experiment.py`)

Evaluates multiple predefined post-processing methods on a single dataset with detailed output.

```bash
python -m postprocess.run_experiment --dataset Cityscapes_sample_with_unbiased_annotations
```

**Arguments:**
- `--dataset`: Name of the JSON file in `our_datasets/` (without `.json`)
- `--data_dir`: Directory containing dataset files (default: `our_datasets/`)
- `--output_dir`: Directory for plot outputs
- `--alpha`: Dirichlet smoothing parameter (default: 0.5)
- `--skip_plots`: Skip generating histogram plots
- `--analyze_annotations`: Analyze annotation timing (speedup) and damper values
- `--sample_plots`: Generate sample distribution bar charts (optional, for debugging)

**Output includes:**
- Dataset statistics
- Baseline comparison (biased vs unbiased)
- Transition matrix for class blending
- Evaluation results for each predefined method
- Annotation timing analysis and damper values (if `--analyze_annotations` is set)

---

### Automatic Parameter Tuning (`run_tuning.py`)

Searches over a parameter grid to find the best method configuration, then evaluates on a held-out validation set.

```bash
python -m postprocess.run_tuning --dataset Cityscapes_sample_with_unbiased_annotations
```

**Arguments:**
- `--dataset`: Name of the JSON file (required)
- `--train_ratio`: Fraction of data for tuning (default: 0.5)
- `--seed`: Random seed for data split (default: 42)
- `--metric`: Metric to optimize (see below)
- `--maximize`: If set, maximize the metric instead of minimizing
- `--alpha`: Dirichlet smoothing parameter (default: 0.5)
- `--quiet`: Reduce output verbosity
- `--save_config`: Path to save tuning config JSON (e.g., `configs/cityscapes.json`)
- `--use_mlp`: Use MLP-based post-processing (experimental, requires PyTorch)

**Example with different metrics:**
```bash
# Minimize JS divergence (default)
python -m postprocess.run_tuning --dataset Cityscapes_sample_with_unbiased_annotations

# Minimize mean delta (change in divergence)
python -m postprocess.run_tuning --dataset Cityscapes_sample_with_unbiased_annotations \
    --metric delta_summary.mean.point

# Maximize fraction improved
python -m postprocess.run_tuning --dataset Cityscapes_sample_with_unbiased_annotations \
    --metric delta_summary.fraction_improved --maximize
```

---

### Batch Tuning All Datasets (`run_tuning_all.py`)

Batch-runs parameter tuning for all supported datasets and generates config files for later use with `run_apply.py`.

```bash
python -m postprocess.run_tuning_all
```

**Arguments:**
- `--datasets`: Specific datasets to tune (default: all). Options: `cityscapes`, `kitti`, `pascalvoc`, `coco`
- `--skip_coco`: Skip COCO super-category tuning (faster)
- `--fast_smoothing_grid`: Use reduced smoothing parameter grid (faster but less thorough)

**Output:**
- Single configs: `configs/cityscapes.json`, `configs/kitti.json`, `configs/pascalvoc.json`
- Multi-config for COCO: `configs/coco.json` with per-super-category parameters

**Example:**
```bash
# Tune only Cityscapes and Kitti
python -m postprocess.run_tuning_all --datasets cityscapes kitti

# Fast tuning for all datasets
python -m postprocess.run_tuning_all --fast_smoothing_grid
```

---

### Apply Post-Processing (`run_apply.py`)

Apply tuned post-processing parameters to soft label files. Supports both single-dataset configs and COCO multi-configs.

```bash
python -m postprocess.run_apply --config cityscapes --input soft_Cityscapes_train
```

**Arguments:**
- `--config`: Config file path or shortcut (e.g., `cityscapes` → `configs/cityscapes.json`)
- `--input`: Input soft label JSON file (path or name to search in `our_datasets/`)
- `--output`: Path for output JSON file (default: `outputs/application/<input>_postprocessed.json`)
- `--inplace`: Overwrite input file in place instead of creating new output
- `--data_dir`: Directory containing soft label files (default: `our_datasets/`)
- `--stats_output`: Path for statistics JSON (default: `outputs/application/<input>_stats.json`)
- `--plot_output`: Path for histogram plot (default: `outputs/application/<input>_histogram.png`)
- `--skip_plot`: Skip generating histogram plot
- `--quiet`: Reduce output verbosity

**Output structure:**
```
outputs/application/
├── <input_name>_postprocessed.json  # Main output (unless --inplace)
├── <input_name>_stats.json          # Statistics
└── <input_name>_histogram.png       # Before/after histogram
```

**Example:**
```bash
# Apply and overwrite input file
python -m postprocess.run_apply --config pascalvoc --input soft_PascalVOC_2012_segmentation_val --inplace

# Apply COCO multi-config
python -m postprocess.run_apply --config coco --input soft_COCO_2017_val
```

---

### Calibration Visualization (`run_calibration.py`)

Visualize calibration by creating scatter plots comparing reviewer agreement scores against various probability estimates.

```bash
python -m postprocess.run_calibration --dataset cityscapes
```

**Arguments:**
- `--dataset`: Dataset to visualize. Options: `cityscapes`, `kitti`, `pascalvoc`, `coco`
- `--alpha`: Dirichlet smoothing alpha parameter (default: 0.5)
- `--output_dir`: Output directory for plots (default: `outputs/calibration/<dataset>/`)

**Output:**
Scatter plots showing reviewer agreement (X-axis) vs probability estimates (Y-axis) for:
- Raw soft labels
- Unbiased distributions
- Dirichlet-smoothed distributions
- Post-processed distributions

For COCO, merges all super-category files and colors points by super-category.

---

## Understanding the Output

### Example Output Explained

```
Avg P(proposed): unbiased=0.8638, biased=0.9243, postproc=0.9082 -> reduction=-0.0161 (-26.62%)
```

| Value | Meaning |
|-------|---------|
| `unbiased=0.8638` | Ground truth: average probability mass on the proposed class is 86.38% |
| `biased=0.9243` | Input biased data: inflated to 92.43% due to annotator bias toward proposals |
| `postproc=0.9082` | After post-processing: reduced to 90.82% |
| `reduction=-0.0161` | Post-processing moved 1.61% probability mass away from the proposed class |
| `(-26.62%)` | This is 26.62% of the original bias (`biased - unbiased = 0.0605`) |

**Interpretation:** The bias inflated the proposed class by ~6%. Post-processing removed ~27% of that inflation.

---

### Distribution Comparison Metrics

```
Distribution comparison (non-bootstrapped) postprocessed:
  native_kl: inf+-nan, [0.000000]
  epsilon_kl: 0.293459+-1.105735, [0.000000]
  dirichlet_kl: 0.059317+-0.129897, [0.000000]
  dirichlet_js: 0.013368+-0.025670, [0.000000]
  dirichlet_js_sqrt: 0.073327+-0.089393, [0.000000]
```

These measure the **divergence between post-processed and unbiased distributions** (lower = better):

| Metric | Description |
|--------|-------------|
| `native_kl` | Raw KL divergence (often `inf` due to zero probabilities) |
| `epsilon_kl` | KL divergence with epsilon smoothing |
| `dirichlet_kl` | KL divergence with Dirichlet smoothing (more stable) |
| `dirichlet_js` | Jensen-Shannon divergence with Dirichlet smoothing |
| `dirichlet_js_sqrt` | √JS divergence (bounded 0-1, recommended metric) |

**Format:** `mean+-std, [median]` across all objects in the dataset.

**Lower values = post-processed distribution is closer to ground truth.**

---

### Post-Processing Impact (Delta Analysis)

```
Post-processing impact on √JS divergence (Δ = after - before) [bootstrapped]
────────────────────────────────────────────────────────
Mean Δ-divergence:   -0.00088 (95% CI [-0.00018, -0.00161]) -> (-1.20%)
Median Δ-divergence: +0.00000 (95% CI [+0.00000, +0.00000])
Fraction improved:  9.8%
Fraction worsened:  7.6%
```

This compares **before post-processing (biased)** vs **after post-processing**:

| Metric | Meaning |
|--------|---------|
| `Δ-divergence` | `divergence(postproc, unbiased) - divergence(biased, unbiased)` |
| Negative Δ | Post-processing **improved** (got closer to truth) |
| Positive Δ | Post-processing **worsened** (got farther from truth) |
| `95% CI` | Confidence interval from bootstrap resampling |
| `(-1.20%)` | Relative improvement compared to baseline divergence |
| `Fraction improved` | % of objects where post-processing reduced divergence |
| `Fraction worsened` | % of objects where post-processing increased divergence |

**Interpretation of the example:**
- Mean improvement of -0.00088 (distribution got ~1.2% closer to truth)
- Median shows no change (many objects unaffected)
- 9.8% of objects improved, 7.6% worsened, ~82% unchanged

---

## Available Post-Processing Methods

### 1. Identity
No transformation. Useful as baseline.

### 2. Resample
Bootstrap resampling within each object's annotations.

### 3. Downweight
Reduces count of the proposed class by a factor.

### 4. CleverLabel
Sophisticated bias correction with two components:
- **Bias Correction (BC):** Estimates and removes proposal acceptance bias
- **Class Blending (CB):** Smooths predictions using class confusion patterns

**CleverLabel Parameters:**
| Parameter | Description | Range |
|-----------|-------------|-------|
| `delta` | Acceptance rate baseline (higher = more correction) | 0.0 - 0.6 |
| `mu` | BC/CB balance (1.0 = BC only, 0.0 = CB only) | 0.0 - 1.0 |
| `avoid_overcorrection_threshold` | Skip correction if acceptance rate below threshold | 0.0 - 0.9 |

---

## Metric Paths for Tuning

Use `--metric` to specify which value to optimize:

| Metric Path | Description | Optimize |
|-------------|-------------|----------|
| `distribution_comparison.dirichlet_js_sqrt.mean` | Mean √JS divergence | minimize |
| `delta_summary.mean.point` | Mean Δ-divergence | minimize |
| `delta_summary.fraction_improved` | % of objects improved | maximize |
| `avg_prob_proposed.improvement` | Reduction in proposed class bias | maximize |

---

## File Layout

```
postprocess/
├── __init__.py              # Package exports
├── clever_labeling.py       # Core CleverLabel algorithm (single-entry)
├── config_loader.py         # Config loading utilities (single + multi-config)
├── data_loader.py           # Dataset loading, filtering, train/val splitting
├── evaluation.py            # Evaluation utilities + result printing
├── metrics.py               # Divergence metrics (KL, JS, etc.)
├── metrics_comparison.py    # Distribution comparison utilities
├── smoothing.py             # Type definitions + smoothing functions
├── methods.py               # Post-processing methods (probs → probs)
├── methods_wrappers.py      # Smoothing-aware method wrappers (counts → probs)
├── methods_factory.py       # Factory functions + parameter configs
├── mlp_tuning.py            # MLP-based post-processing (experimental)
├── visualization.py         # Plotting + DataFrame utilities
├── run_experiment.py        # Detailed single-dataset evaluation
├── run_tuning.py            # Automatic parameter tuning with train/val split
├── run_tuning_all.py        # Batch tuning for all datasets
├── run_apply.py             # Apply tuned method to full dataset
└── run_calibration.py       # Calibration visualization
```

---

## Architecture: How The Files Connect

The post-processing pipeline transforms raw annotation counts into corrected probability distributions. Here's how the modules connect:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   counts (Dataset)                                                          │
│       │                                                                     │
│       ▼                                                                     │
│   ┌──────────────────────────┐                                              │
│   │  methods_wrappers.py     │  Wraps methods with smoothing pipelines      │
│   │  create_method_with_     │                                              │
│   │  smoothing()             │                                              │
│   └──────────┬───────────────┘                                              │
│              │                                                              │
│              │  DIRICHLET_POST: counts → normalize → probs → PP → counts → Dirichlet → probs
│              │  DIRICHLET_PRE:  counts → Dirichlet → probs → PP → probs     │
│              │  EPSILON:        counts → normalize → probs → PP → epsilon → probs
│              ▼                                                              │
│   ┌──────────────────────────┐                                              │
│   │  methods.py              │  Core PP algorithms (probs → probs)          │
│   │  - identity()            │                                              │
│   │  - perfect()             │                                              │
│   │  - apply_cleverlabel()   │───────┐                                      │
│   │  - downweight_proposed() │       │                                      │
│   │  - resample()            │       │                                      │
│   └──────────────────────────┘       │                                      │
│                                      ▼                                      │
│                         ┌──────────────────────────┐                        │
│                         │  clever_labeling.py      │ CleverLabel algorithm  │
│                         │  cleverlabel()           │ for single entry       │
│                         └──────────────────────────┘                        │
│                                                                             │
│   probs (DistDataset)                                                       │
│       ▲                                                                     │
│       │                                                                     │
│   ┌──────────────────────────┐                                              │
│   │  smoothing.py            │  Type definitions + smoothing functions      │
│   │  - Smoothing enum        │  - counts_to_dirichlet_probs()               │
│   │  - Counts, Dataset       │  - epsilon_smooth_probs()                    │
│   │  - Distribution          │  - probs_to_counts()                         │
│   │  - DistDataset           │  - mle_probs()                               │
│   └──────────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| File | Responsibility | Input → Output |
|------|----------------|----------------|
| `smoothing.py` | Type aliases (Counts, Dataset, Distribution, DistDataset), Smoothing enum, and all smoothing functions (Dirichlet, epsilon, MLE, probs↔counts conversion) | Various |
| `clever_labeling.py` | Core CleverLabel algorithm for **one entry**: Bias Correction (BC) and Class Blending (CB) | single probs/counts → single probs/counts |
| `methods.py` | Post-processing methods applied to **full datasets**: identity, perfect, cleverlabel, downweight, resample | DistDataset → DistDataset |
| `methods_wrappers.py` | Wraps `methods.py` functions with smoothing pipelines to create final methods | Dataset (counts) → DistDataset (probs) |
| `methods_factory.py` | Factory functions to instantiate methods with parameters. Contains `DEFAULT_METHOD_CONFIGS` and `DEFAULT_SMOOTHING_VARIANTS` | config → callable method |
| `metrics.py` | Divergence metrics: KL, JS, JS√ on probability distributions | Distribution × Distribution → float |
| `metrics_comparison.py` | Compare two datasets using multiple metrics | DistDataset × DistDataset → summary dict |
| `evaluation.py` | High-level evaluation: run methods, compute metrics, format output | method + data → results |

### Example: How a Method Gets Created

```python
# 1. Define method config (methods_factory.py)
config = ("cleverlabel", {"delta": 0.1, "mu": 0.75})

# 2. Create method with smoothing pipeline (methods_factory.py → methods_wrappers.py)
# New path: counts → probs
methods = get_default_methods(proposed, classes, transition_c, unbiased)
# This calls create_method_with_smoothing() which:
#   a) Creates a probs→probs method from methods.py
#   b) Wraps it with the chosen smoothing pipeline

# Legacy path: counts → counts
methods_legacy = get_default_methods_legacy(proposed, classes, transition_c, unbiased)

# 3. The resulting method transforms counts → probs (new) or counts → counts (legacy)
postprocessed_probs = method(biased_counts)
```

### Smoothing Pipelines

The `Smoothing` enum (in `smoothing.py`) defines three pipelines:

| Pipeline | Flow | Use Case |
|----------|------|----------|
| `DIRICHLET_POST` | counts → normalize → probs → PP → pseudo-counts → Dirichlet → probs | Standard: smooth after PP |
| `DIRICHLET_PRE` | counts → Dirichlet → probs → PP → probs | Alternative: smooth before PP |
| `EPSILON` | counts → normalize → probs → PP → epsilon-smooth → probs | Minimal smoothing |

### Core Modules

| File | Purpose |
|------|---------|
| `clever_labeling.py` | Implements the CleverLabel bias correction algorithm with Bias Correction (BC) and Class Blending (CB) |
| `metrics.py` | Divergence metrics: KL divergence, Jensen-Shannon divergence |
| `data_loader.py` | Loads datasets from JSON files, handles train/validation splitting |
| `methods_factory.py` | Factory functions for all post-processing methods + parameter grid definitions |
| `evaluation.py` | Computes and prints evaluation metrics |
