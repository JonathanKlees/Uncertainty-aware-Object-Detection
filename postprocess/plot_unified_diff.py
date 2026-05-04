#!/usr/bin/env python3
"""Unified line plot of biased-minus-unbiased probability differences across datasets."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from postprocess.data_loader import (
    load_dataset_from_json, filter_to_common_keys,
    create_distributions, apply_cantsolve_multiplier,
    resolve_dataset_name, find_dataset_path,
)
from postprocess.visualization import build_comparison_dataframe

data_dir = "our_datasets/unbiased_samples"

datasets = {
    "Cityscapes": "cityscapes",
    "KITTI": "kitti",
    "Pascal VOC": "pascalvoc",
    "COCO Kitchen": "coco_kitchen",
    "COCO Animal": "coco_animal",
}

colors = {
    "Cityscapes": "#1f77b4",
    "KITTI": "#ff7f0e",
    "Pascal VOC": "#2ca02c",
    "COCO Kitchen": "#d62728",
    "COCO Animal": "#9467bd",
}

# Same custom bins as plot_probability_difference, but without the fine bins near zero
bins = np.array(
    [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
    + [0.0]
    + [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Threshold: drop objects with near-zero difference
ZERO_THRESHOLD = 0.01
bin_centers = 0.5 * (bins[:-1] + bins[1:])

fig, ax = plt.subplots(figsize=(10, 5))

for label, shortcut in datasets.items():
    dataset_name = resolve_dataset_name(shortcut)
    dataset_path = find_dataset_path(data_dir, dataset_name)

    biased_c, unbiased_c, proposed_c, classes = load_dataset_from_json(dataset_path)
    biased_c, unbiased_c, proposed_c = filter_to_common_keys(biased_c, unbiased_c, proposed_c)
    unbiased_c = apply_cantsolve_multiplier(unbiased_c)

    unbiased_d = create_distributions(unbiased_c)
    biased_d = create_distributions(biased_c)

    df = build_comparison_dataframe(unbiased_d, biased_d, proposed_c)
    diffs = (df["biased"] - df["unbiased"]).values

    # Drop near-zero differences to avoid dominating spike
    n_total = len(diffs)
    diffs = diffs[np.abs(diffs) >= ZERO_THRESHOLD]
    n_dropped = n_total - len(diffs)
    print(f"  {label}: {n_total} objects, dropped {n_dropped} with |diff| < {ZERO_THRESHOLD} ({100*n_dropped/n_total:.1f}%)")

    counts, _ = np.histogram(diffs, bins=bins)
    density = counts / counts.sum()  # normalize for comparability

    c = colors[label]
    ax.plot(bin_centers, density, color=c, linewidth=2, label=label, zorder=3)
    ax.fill_between(bin_centers, 0, density, color=c, alpha=0.15, zorder=2)

ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1, zorder=1)
ax.set_xlabel("Probability Difference (biased \u2212 unbiased)", fontsize=12)
ax.set_ylabel("Fraction of Objects", fontsize=12)
ax.set_title("Distribution of Annotation Bias Across Datasets", fontsize=13)
ax.legend(fontsize=11, framealpha=0.9)
ax.set_xlim(-0.55, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = "outputs/paper/unified_hist_diff_biased_minus_unbiased.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200)
plt.close()
print(f"Saved: {out}")
