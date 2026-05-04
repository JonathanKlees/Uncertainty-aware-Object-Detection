"""
Visualization utilities for plotting distributions and analysis results.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from postprocess.metrics import Metric
from postprocess.metrics_comparison import DivergenceRecord


def plot_probability_histograms(
    unbiased_probs: List[float],
    biased_probs: List[float],
    output_path: str,
    num_bins: int = 25,
) -> None:
    """
    Plot side-by-side histograms of unbiased vs biased probabilities.
    """
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(unbiased_probs, bins=num_bins, alpha=0.7, color="skyblue")
    plt.title("Unbiased: P(proposed class)")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.hist(biased_probs, bins=num_bins, alpha=0.7, color="orange")
    plt.title("Biased: P(proposed class)")
    plt.xlabel("Probability")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probability_difference(
    unbiased_probs: List[float],
    biased_probs: List[float],
    output_path: str,
) -> None:
    """
    Plot histogram of probability differences (biased - unbiased).
    """
    diffs = [b - u for u, b in zip(unbiased_probs, biased_probs)]
    
    # Custom bins: fine resolution near 0, coarser for tails
    bins = (
        [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.01, -0.005]
        + [0.0]
        + [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    
    plt.figure(figsize=(8, 4))
    plt.hist(diffs, bins=bins, alpha=0.8, color="purple", edgecolor="black")
    plt.title("Difference: biased_proposed - unbiased_proposed")
    plt.xlabel("Probability Difference")
    plt.ylabel("Count")
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_transition_matrix(
    classes: List[str],
    transition_c: Dict[str, List[float]],
) -> None:
    """Print the transition matrix for class blending."""
    print("\nTransition Matrix (Class Blending)")
    print("===================================")

    col_width = max(max(len(c) for c in classes), 10)
    
    # Header
    header = " " * (col_width + 2)
    for c in classes:
        header += f"{c:>{col_width}}  "
    print(header)

    # Rows
    for row_class in classes:
        row = transition_c[row_class]
        line = f"{row_class:>{col_width}}  "
        for value in row:
            line += f"{value:>{col_width}.4f}  "
        print(line)


def print_filtered_examples(
    df: pd.DataFrame,
    threshold: float = 0.3,
    max_rows: int = 20,
) -> None:
    """Print filtered examples where unbiased and biased differ significantly."""
    # Filter where unbiased > biased + threshold
    filtered_high = df[df["unbiased"] > df["biased"] + threshold]
    print(f"\nTotal rows: {len(df)}, Filtered (unbiased > biased + {threshold}): {len(filtered_high)}")
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', None, 'display.width', 120):
        print(filtered_high.head(max_rows).to_string(index=False))

    # Filter where unbiased + threshold < biased 
    filtered_low = df[df["unbiased"] + threshold < df["biased"]]
    print(f"\nTotal rows: {len(df)}, Filtered (unbiased + {threshold} < biased): {len(filtered_low)}")
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', None, 'display.width', 120):
        print(filtered_low.head(max_rows).to_string(index=False))


def build_comparison_dataframe(
    map_id_unbiased_dist: Dict[str, Dict[str, float]],
    map_id_biased_dist: Dict[str, Dict[str, float]],
    map_id_proposed_class: Dict[str, str],
) -> pd.DataFrame:
    """
    Build a DataFrame comparing unbiased and biased probabilities for each object.
    
    Returns DataFrame with columns: id, proposed_class, unbiased, biased
    """
    rows = []
    key_mismatch_count = 0
    norm_issue_count = 0
    
    for obj_id in map_id_unbiased_dist:
        unbiased_dist = map_id_unbiased_dist.get(obj_id)
        biased_dist = map_id_biased_dist.get(obj_id)
        proposed_cls = map_id_proposed_class.get(obj_id)
        
        if unbiased_dist is None or biased_dist is None or proposed_cls is None:
            continue
        
        # Sanity: same label space
        if set(unbiased_dist.keys()) != set(biased_dist.keys()):
            key_mismatch_count += 1
        
        # Sanity: normalized distributions
        if not (
            abs(sum(unbiased_dist.values()) - 1.0) < 1e-6
            and abs(sum(biased_dist.values()) - 1.0) < 1e-6
        ):
            norm_issue_count += 1
        
        unbiased_prob = float(unbiased_dist.get(proposed_cls, 0.0))
        biased_prob = float(biased_dist.get(proposed_cls, 0.0))
        
        rows.append({
            "id": obj_id,
            "proposed_class": proposed_cls,
            "unbiased": unbiased_prob,
            "biased": biased_prob,
        })
    
    if key_mismatch_count or norm_issue_count:
        print(f"Sanity warnings — key mismatches: {key_mismatch_count}, normalization issues: {norm_issue_count}")
    
    df = (
        pd.DataFrame(rows, columns=["id", "proposed_class", "unbiased", "biased"])
        .sort_values(by=["unbiased"], ascending=False)
        if rows
        else pd.DataFrame(columns=["id", "proposed_class", "unbiased", "biased"])
    )
    
    return df


def plot_sample_distributions(
    records: Dict[str, DivergenceRecord],
    unbiased_probs: Dict[str, Dict[str, float]],
    biased_probs: Dict[str, Dict[str, float]],
    postprocessed_probs: Dict[str, Dict[str, float]],
    map_id_proposed_class: Dict[str, str],
    output_path: str,
    num_samples: int = 10,
    metric: Metric = Metric.JS_SQRT,
    seed: int = 42,
) -> None:
    """
    Plot bar charts comparing distributions for sample objects.
    
    Shows:
    - Top num_samples objects where divergence improved most
    - Top num_samples objects where divergence worsened most
    - num_samples random objects
    
    Args:
        records: Pre-computed divergence records from compute_all_divergences()
        unbiased_probs: Ground truth distributions
        biased_probs: Biased distributions (before post-processing)
        postprocessed_probs: Post-processed distributions
        map_id_proposed_class: Proposed class per object
        output_path: Path to save the plot
        num_samples: Number of samples for each category (default: 10)
        metric: Divergence metric (default: Metric.JS_SQRT)
        seed: Random seed for reproducibility
    """
    # Extract deltas from pre-computed records
    deltas = []  # (obj_id, delta, before_div, after_div)
    for obj_id, rec in records.items():
        delta = rec.delta(metric)
        before_div = rec.biased(metric)
        after_div = rec.postproc(metric)
        deltas.append((obj_id, delta, before_div, after_div))
    
    # Sort by delta
    deltas.sort(key=lambda x: x[1])
    
    # Select samples
    improved = deltas[:num_samples]  # Most negative delta = most improved
    worsened = deltas[-num_samples:][::-1]  # Most positive delta = most worsened
    
    # Random samples (excluding already selected)
    selected_ids = {x[0] for x in improved + worsened}
    remaining = [x for x in deltas if x[0] not in selected_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    random_samples = remaining[:num_samples]
    
    all_samples = [
        ("IMPROVED", improved),
        ("WORSENED", worsened),
        ("RANDOM", random_samples),
    ]
    
    # Get all classes (sorted)
    sample_id = next(iter(unbiased_probs))
    classes = sorted(unbiased_probs[sample_id].keys())
    
    # Create figure with subplots
    n_categories = len(all_samples)
    n_per_category = num_samples
    
    fig, axes = plt.subplots(
        n_categories, n_per_category, 
        figsize=(3 * n_per_category, 4 * n_categories),
        squeeze=False
    )
    
    bar_width = 0.25
    x = np.arange(len(classes))
    
    for cat_idx, (category_name, samples) in enumerate(all_samples):
        for sample_idx, (obj_id, delta, before_div, after_div) in enumerate(samples):
            if sample_idx >= n_per_category:
                break
            
            ax = axes[cat_idx, sample_idx]
            
            unbiased = unbiased_probs[obj_id]
            biased = biased_probs[obj_id]
            postproc = postprocessed_probs[obj_id]
            proposed = map_id_proposed_class.get(obj_id, "?")
            
            # Get probabilities in class order
            unbiased_vals = [unbiased.get(c, 0) for c in classes]
            biased_vals = [biased.get(c, 0) for c in classes]
            postproc_vals = [postproc.get(c, 0) for c in classes]
            
            # Plot bars
            ax.bar(x - bar_width, unbiased_vals, bar_width, label='Unbiased', color='green', alpha=0.7)
            ax.bar(x, biased_vals, bar_width, label='Biased', color='orange', alpha=0.7)
            ax.bar(x + bar_width, postproc_vals, bar_width, label='Postproc', color='blue', alpha=0.7)
            
            # Highlight proposed class
            proposed_idx = classes.index(proposed) if proposed in classes else -1
            if proposed_idx >= 0:
                ax.axvline(x=proposed_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Labels
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=6)
            ax.set_ylim(0, 1.05)
            
            # Title with delta info
            delta_sign = "+" if delta >= 0 else ""
            title = f"{category_name}\nΔ={delta_sign}{delta:.4f}\n(before={before_div:.3f}, after={after_div:.3f})\nproposed={proposed}"
            ax.set_title(title, fontsize=8)
            
            if sample_idx == 0:
                ax.set_ylabel('Probability')
            
            if cat_idx == 0 and sample_idx == 0:
                ax.legend(fontsize=6, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample distribution plots saved to: {output_path}")


# =============================================================================
# Annotation Timing & Damper Visualization
# =============================================================================

def plot_annotation_times(
    biased_times: List[float],
    unbiased_times: List[float],
    output_path: str,
    max_time: float = 15.0,
) -> None:
    """
    Plot annotation time distributions for biased vs unbiased conditions.
    
    Creates:
    - Overlapping histograms
    - Box plot comparison
    
    Args:
        biased_times: List of annotation times with proposal shown
        unbiased_times: List of annotation times without proposal
        output_path: Path to save the figure
        max_time: Maximum time to show on x-axis (clips outliers)
    """
    # Clip to max_time for visualization
    biased_clipped = [min(t, max_time) for t in biased_times]
    unbiased_clipped = [min(t, max_time) for t in unbiased_times]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1 = axes[0]
    bins = np.linspace(0, max_time, 50)
    ax1.hist(biased_clipped, bins=bins, alpha=0.6, label=f'Biased (n={len(biased_times)})', 
             color='orange', density=True)
    ax1.hist(unbiased_clipped, bins=bins, alpha=0.6, label=f'Unbiased (n={len(unbiased_times)})', 
             color='blue', density=True)
    ax1.axvline(np.median(biased_times), color='orange', linestyle='--', linewidth=2, 
                label=f'Biased median={np.median(biased_times):.2f}s')
    ax1.axvline(np.median(unbiased_times), color='blue', linestyle='--', linewidth=2,
                label=f'Unbiased median={np.median(unbiased_times):.2f}s')
    ax1.set_xlabel('Annotation Time (seconds)')
    ax1.set_ylabel('Density')
    ax1.set_title('Annotation Time Distribution')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, max_time)
    
    # Box plot
    ax2 = axes[1]
    box_data = [biased_times, unbiased_times]
    bp = ax2.boxplot(box_data, labels=['Biased\n(with proposal)', 'Unbiased\n(no proposal)'],
                     showfliers=False)  # Hide outliers for clarity
    ax2.set_ylabel('Annotation Time (seconds)')
    ax2.set_title('Annotation Time Comparison')
    
    # Add mean markers
    means = [np.mean(biased_times), np.mean(unbiased_times)]
    ax2.scatter([1, 2], means, color='red', marker='D', s=50, zorder=3, label='Mean')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Annotation time plot saved to: {output_path}")


def plot_damper_histogram(
    damper_values: List[float],
    output_path: str,
    max_damper: float = 2.0,
    num_bins: int = 40,
) -> None:
    """
    Plot histogram of damper values (unbiased_prob / biased_prob).
    
    biased * damper = unbiased
    - damper < 1: proposal inflated probability
    - damper > 1: proposal deflated probability
    
    Args:
        damper_values: List of damper values
        output_path: Path to save the figure
        max_damper: Maximum damper value to show (clips x-axis at this value)
        num_bins: Number of histogram bins
    """
    # Clip values for visualization
    clipped = [min(d, max_damper) for d in damper_values]
    
    plt.figure(figsize=(8, 5))
    
    bins = np.linspace(0, max_damper, num_bins)
    n, _, patches = plt.hist(clipped, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Color bins based on damper value
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 1.0:
            patch.set_facecolor('orange')  # Proposal inflated probability
        elif bin_center > 1.0:
            patch.set_facecolor('green')  # Proposal deflated probability
        else:
            patch.set_facecolor('gray')  # Neutral
    
    # Add vertical lines for statistics
    mean_val = np.mean(damper_values)
    median_val = np.median(damper_values)
    
    plt.axvline(1.0, color='black', linestyle='-', linewidth=2, label='No effect (1.0)')
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean={mean_val:.3f}')
    plt.axvline(median_val, color='blue', linestyle=':', linewidth=2,
                label=f'Median={median_val:.3f}')
    
    plt.xlabel('Damper (unbiased / biased)')
    plt.ylabel('Count')
    plt.title(f'Damper Distribution (n={len(damper_values)}, clipped at {max_damper})\n<1 = proposal inflated prob, >1 = proposal deflated prob')
    plt.legend()
    plt.xlim(0, max_damper)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Damper histogram saved to: {output_path}")


def plot_damper_per_class(
    per_class_dampers: Dict[str, List[float]],
    output_path: str,
    max_damper: float = 2.0,
) -> None:
    """
    Plot damper distribution per proposed class as box plots.
    
    Args:
        per_class_dampers: Dict mapping class name -> list of damper values
        output_path: Path to save the figure
        max_damper: Maximum damper value to show (clips outliers)
    """
    # Sort classes by count
    sorted_classes = sorted(per_class_dampers.items(), key=lambda x: len(x[1]), reverse=True)
    
    if not sorted_classes:
        print("No per-class damper data available")
        return
    
    # Clip values
    clipped_data = []
    labels = []
    for cls, dampers in sorted_classes:
        clipped = [min(d, max_damper) for d in dampers]
        clipped_data.append(clipped)
        labels.append(f"{cls}\n(n={len(dampers)})")
    
    plt.figure(figsize=(max(10, len(sorted_classes) * 0.8), 6))
    
    bp = plt.boxplot(clipped_data, labels=labels, showfliers=False, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.tab10.colors
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
    
    # Add horizontal line at 1.0 (no effect)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='No effect (1.0)')
    
    # Add mean markers
    means = [np.mean(dampers) for _, dampers in sorted_classes]
    x_positions = range(1, len(means) + 1)
    plt.scatter(x_positions, [min(m, max_damper) for m in means], 
                color='red', marker='D', s=50, zorder=3, label='Mean')
    
    plt.ylabel('Damper (unbiased / biased)')
    plt.xlabel('Proposed Class')
    plt.title(f'Damper Distribution by Proposed Class (clipped at {max_damper})\n<1 = proposal inflated prob')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max_damper + 0.1)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class damper plot saved to: {output_path}")
