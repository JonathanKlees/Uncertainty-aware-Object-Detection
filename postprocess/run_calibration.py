#!/usr/bin/env python3
"""
Visualize human calibration data from unbiased_samples/.

Creates X-Y scatter plots:
- X-axis: review_results (reviewer agreement scores, 0-1)
- Y-axis: probability estimates (soft_label, unbiased, dirichlet_smoothed, postprocessed)
- Color: drawn_class_name (non-COCO) or super-category (COCO)

Usage:
    # Single dataset (color by drawn_class_name)
    python -m postprocess.run_calibration --dataset pascalvoc --alpha 0.5

    # COCO (merges all super-category files, color by super-category)
    python -m postprocess.run_calibration --dataset coco --alpha 0.5

    # With post-processing config
    python -m postprocess.run_calibration --dataset pascalvoc --config configs/pascalvoc.json
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from postprocess.calibration_utils import (
    ObjectCalibrationData,
    load_dataset,
    load_default_config,
    extract_object_calibration_data,
    classify_review_bucket,
    get_default_data_dir,
    get_default_aggregated_dir,
    load_aggregated_index,
)


@dataclass
class CalibrationPoint:
    """Single calibration data point."""
    review_score: float  # X-axis: reviewer agreement (0-1)
    soft_label_prob: float  # Y: biased probability
    unbiased_prob: float  # Y: ground truth probability
    dirichlet_prob: float  # Y: Dirichlet-smoothed unbiased
    postprocessed_prob: Optional[float]  # Y: post-processed (if config available)
    postprocessed_selected_prob: Optional[float]  # Y: post-processed selected (if available)
    color_label: str  # For coloring: drawn_class_name or super-category
    drawn_class_name: str  # Original drawn class
    proposed_class: str  # Proposed class from object
    majority_class: str  # Majority class from unbiased distribution
    
    def get_match_type(self) -> str:
        """Get match type for coloring: 'proposed', 'majority', or 'other'."""
        if self.drawn_class_name == self.proposed_class:
            return "proposed"
        elif self.drawn_class_name == self.majority_class:
            return "majority"
        else:
            return "other"


def convert_to_calibration_points(
    object_data_list: List[ObjectCalibrationData],
) -> List[CalibrationPoint]:
    """
    Convert ObjectCalibrationData objects to CalibrationPoint objects.
    
    Expands each object into multiple CalibrationPoints (one per review_result).
    """
    points = []
    for obj_data in object_data_list:
        for review_score in obj_data.review_results:
            points.append(CalibrationPoint(
                review_score=review_score,
                soft_label_prob=obj_data.biased_prob,
                unbiased_prob=obj_data.unbiased_prob,
                dirichlet_prob=obj_data.dirichlet_prob,
                postprocessed_prob=obj_data.postprocessed_prob,
                postprocessed_selected_prob=obj_data.postprocessed_selected_prob,
                color_label=obj_data.color_label,
                drawn_class_name=obj_data.drawn_class,
                proposed_class=obj_data.proposed_class,
                majority_class=obj_data.unbiased_majority_class,
            ))
    return points


def plot_scatter(
    points: List[CalibrationPoint],
    prob_type: str,
    output_path: str,
    title: str,
    jitter_amount: float = 0.05,
) -> None:
    """
    Create X-Y scatter plot for a specific probability type.
    
    Args:
        points: List of calibration points
        prob_type: One of "soft_label", "unbiased", "dirichlet", "postprocessed"
        output_path: Path to save plot
        title: Plot title
        jitter_amount: Amount of random jitter to add to x-values 
    """
    # Get probability values
    if prob_type == "soft_label":
        y_values = [p.soft_label_prob for p in points]
    elif prob_type == "unbiased":
        y_values = [p.unbiased_prob for p in points]
    elif prob_type == "dirichlet":
        y_values = [p.dirichlet_prob for p in points]
    elif prob_type == "postprocessed":
        y_values = [p.postprocessed_prob for p in points if p.postprocessed_prob is not None]
        points = [p for p in points if p.postprocessed_prob is not None]
    elif prob_type == "postprocessed_selected":
        y_values = [p.postprocessed_selected_prob for p in points if p.postprocessed_selected_prob is not None]
        points = [p for p in points if p.postprocessed_selected_prob is not None]
    else:
        raise ValueError(f"Unknown prob_type: {prob_type}")
    
    if not points:
        print(f"Warning: No data points for {prob_type}, skipping plot")
        return
    
    x_values = [p.review_score for p in points]
    color_labels = [p.color_label for p in points]
    
    # Calculate Pearson correlation
    pearson_r, p_value = stats.pearsonr(x_values, y_values)
    
    # Add jitter to x-values to separate overlapping points
    jitter = np.random.uniform(-jitter_amount, jitter_amount, len(x_values))
    x_jittered = [x + j for x, j in zip(x_values, jitter)]
    y_jittered = [y + np.random.uniform(-jitter_amount, jitter_amount) for y in y_values]
    
    # Create color mapping
    unique_labels = sorted(set(color_labels))
    cmap = plt.colormaps.get_cmap('tab20').resampled(len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in color_labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with jittered x-values
    scatter = ax.scatter(x_jittered, y_jittered, c=colors, alpha=0.6, s=30, edgecolors='none')
    
    # Add diagonal line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    # Labels and title
    ax.set_xlabel('Review Score (Reviewer Agreement)', fontsize=12)
    ax.set_ylabel(f'Probability ({prob_type})', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add Pearson r annotation
    ax.text(0.02, 0.98, f"Pearson r = {pearson_r:.3f}\np = {p_value:.2e}",
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend for color labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[label],
                          markersize=8, label=label) for label in unique_labels]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_bucket_scatter(
    points: List[CalibrationPoint],
    prob_type: str,
    output_path: str,
    title: str,
    bucket_thresholds: Tuple[float, float] = (0.25, 0.75),
    jitter_x: float = 0.05,
    jitter_y: float = 0.2,
) -> None:
    """
    Create bucket-based scatter plot for a single distribution type.
    
    X-axis: probability values
    Y-axis: review score buckets (no, maybe, clearly)
    Overlay: mean±std (dot + error bar) and median (black star) with values
    Color: drawn_class_name or super-category
    
    Args:
        points: List of calibration points
        prob_type: One of "soft_label", "unbiased", "dirichlet", "postprocessed"
        output_path: Path to save plot
        title: Plot title
        bucket_thresholds: (no_threshold, clearly_threshold) for bucketing
        jitter_x: Amount of jitter on x-axis (probability)
        jitter_y: Amount of jitter on y-axis (bucket position)
    """
    # Get probability values
    if prob_type == "soft_label":
        prob_values = [p.soft_label_prob for p in points]
    elif prob_type == "unbiased":
        prob_values = [p.unbiased_prob for p in points]
    elif prob_type == "dirichlet":
        prob_values = [p.dirichlet_prob for p in points]
    elif prob_type == "postprocessed":
        prob_values = [p.postprocessed_prob for p in points if p.postprocessed_prob is not None]
        points = [p for p in points if p.postprocessed_prob is not None]
    elif prob_type == "postprocessed_selected":
        prob_values = [p.postprocessed_selected_prob for p in points if p.postprocessed_selected_prob is not None]
        points = [p for p in points if p.postprocessed_selected_prob is not None]
    else:
        raise ValueError(f"Unknown prob_type: {prob_type}")
    
    if not points:
        print(f"Warning: No data points for {prob_type}, skipping bucket plot")
        return
    
    # Define buckets
    bucket_names = ["no", "maybe", "clearly"]
    bucket_y_pos = {"no": 0, "maybe": 1, "clearly": 2}
    bucket_colors_bg = {"no": "#ffcccc", "maybe": "#ffffcc", "clearly": "#ccffcc"}
    
    # Color mapping for match types
    match_type_colors = {
        "proposed": "#2ca02c",  # Green - matches proposed class
        "majority": "#1f77b4",  # Blue - matches majority of distribution
        "other": "#d62728",     # Red - neither
    }
    
    # Classify points into buckets
    point_buckets = [classify_review_bucket(p.review_score, bucket_thresholds) for p in points]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect data for statistics
    stats_data = {bucket: [] for bucket in bucket_names}
    
    # Plot scatter points
    for i, (p, prob, bucket) in enumerate(zip(points, prob_values, point_buckets)):
        match_type = p.get_match_type()
        color = match_type_colors[match_type]
        
        # Calculate position with jitter
        x = prob + np.random.uniform(-jitter_x, jitter_x)
        y = bucket_y_pos[bucket] + np.random.uniform(-jitter_y, jitter_y)
        
        ax.scatter(x, y, c=[color], alpha=0.5, s=25, edgecolors='none')
        
        # Collect for statistics
        stats_data[bucket].append(prob)
    
    # Add background shading for buckets
    for bucket in bucket_names:
        y_pos = bucket_y_pos[bucket]
        ax.axhspan(y_pos - 0.4, y_pos + 0.4, alpha=0.15, 
                  color=bucket_colors_bg[bucket], zorder=0)
    
    # Add statistics overlay with values
    for bucket in bucket_names:
        values = stats_data[bucket]
        if not values:
            continue
        
        y_pos = bucket_y_pos[bucket]
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        n_points = len(values)
        
        # Plot mean with error bar (black dot)
        ax.errorbar(mean_val, y_pos, xerr=std_val, fmt='ko', markersize=10, 
                   capsize=5, capthick=2, elinewidth=2, zorder=10)
        
        # Plot median (black star)
        ax.scatter(median_val, y_pos, marker='*', c='black', s=200, 
                  zorder=11, edgecolors='white', linewidths=0.5)
        
        # Add text with actual values
        text_y_offset = 0.35
        ax.text(mean_val, y_pos + text_y_offset, 
               f"mean={mean_val:.3f}±{std_val:.3f}",
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.text(median_val, y_pos - text_y_offset,
               f"median={median_val:.3f} (n={n_points})",
               ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Set y-axis labels (buckets)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(bucket_names, fontsize=11)
    
    # Labels and title
    ax.set_xlabel(f'Probability ({prob_type})', fontsize=12)
    ax.set_ylabel('Review Score Bucket', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.6, 2.6)
    
    # Legend for match type colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=match_type_colors[mt],
                          markersize=8, label=f"{mt} class") for mt in ["proposed", "majority", "other"]]
    # Add legend entries for stats
    handles.append(plt.Line2D([0], [0], marker='o', color='black', markersize=8, label='Mean ± Std'))
    handles.append(plt.Line2D([0], [0], marker='*', color='black', markersize=12, label='Median'))
    
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_review_histogram(
    points: List[CalibrationPoint],
    output_path: str,
    title: str,
) -> None:
    """
    Create histogram of review scores distribution.
    
    Args:
        points: List of calibration points
        output_path: Path to save plot
        title: Plot title
    """
    x_values = [p.review_score for p in points]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(x_values, bins=11, alpha=0.7, color='steelblue', edgecolor='black')
        
    ax.set_xlabel('Review Score (Reviewer Agreement)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize human calibration data"
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cityscapes", "kitti", "pascalvoc", "coco"],
        help="Dataset to load (coco merges all super-category files)"
    )
    parser.add_argument(
        "--data_dir",
        default=get_default_data_dir(),
        help="Directory containing unbiased_samples/"
    )
    parser.add_argument(
        "--source",
        choices=["unbiased", "aggregated"],
        default="aggregated",
        help="Data source for biased/postprocessed distributions. "
             "'unbiased': compute on-the-fly from unbiased_samples. "
             "'aggregated': load from final aggregated files (soft_*.json)."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet smoothing alpha parameter (default: 0.5)"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for plots (default: outputs/calibration/<dataset>)"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "calibration", args.dataset
        )
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"CALIBRATION VISUALIZATION: {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"Source: {args.source}")
    print(f"Alpha: {args.alpha}")
    print(f"Output: {args.output_dir}")
    
    # Load dataset using shared utility
    print(f"\nLoading {args.dataset} dataset...")
    data, classes, is_coco, obj_to_super_category = load_dataset(args.data_dir, args.dataset)
    print(f"  Images: {len(data.get('objects', {}))}")
    print(f"  Classes: {len(classes)}")
    
    # Load aggregated index if requested
    aggregated_index = None
    if args.source == "aggregated":
        print(f"\nLoading aggregated index...")
        aggregated_index = load_aggregated_index(get_default_aggregated_dir(), args.dataset)
    
    # Load config using shared utility
    print(f"\nLoading config...")
    config, multi_config = load_default_config(args.dataset)
    if config:
        print(f"  Method: {config.get('parameters', {}).get('method', 'unknown')}")
        if multi_config:
            print(f"  Multi-config with {len(multi_config.get('super_category_configs', {}))} super-categories")
    else:
        print("  No config loaded")
        
    # Extract calibration data using shared utility
    print("\nExtracting calibration data...")
    object_data_list = extract_object_calibration_data(
        data=data,
        classes=classes,
        alpha=args.alpha,
        is_coco=is_coco,
        obj_to_super_category=obj_to_super_category,
        config=config,
        multi_config=multi_config,
        aggregated_index=aggregated_index,
    )
    print(f"  Objects with calibration: {len(object_data_list)}")
    
    # Convert to CalibrationPoints for plotting
    points = convert_to_calibration_points(object_data_list)
    print(f"  Total points (expanded): {len(points)}")
    
    if not points:
        print("No calibration points found. Ensure dataset has 'human_calibration' field.")
        return
    
    # Generate plots
    # print("\nGenerating scatter plots...")
    
    # # Individual probability type plots
    # for prob_type in ["soft_label", "unbiased", "dirichlet"]:
    #     output_path = os.path.join(args.output_dir, f"scatter_{prob_type}.png")
    #     title = f"{args.dataset.upper()}: Review Score vs {prob_type.replace('_', ' ').title()} Probability"
    #     plot_scatter(points, prob_type, output_path, title)
    
    # # Post-processed plot (if config provided)
    # if config:
    #     output_path = os.path.join(args.output_dir, "scatter_postprocessed.png")
    #     title = f"{args.dataset.upper()}: Review Score vs Post-processed Probability"
    #     plot_scatter(points, "postprocessed", output_path, title)
    
    
    # # Histogram of review scores
    # output_path = os.path.join(args.output_dir, "histogram_review_scores.png")
    # title = f"{args.dataset.upper()}: Distribution of Review Scores"
    # plot_review_histogram(points, output_path, title)
    
    # Bucket scatter plots per distribution type
    print(f"\n{'='*60}")
    print("GENERATING BUCKET PLOTS PER DISTRIBUTION")
    print(f"{'='*60}")
    
    prob_types = ["soft_label", "unbiased", "dirichlet"]
    if config is not None:
        prob_types.append("postprocessed")
        prob_types.append("postprocessed_selected")
    
    for prob_type in prob_types:
        output_path = os.path.join(args.output_dir, f"bucket_{args.source}_{prob_type}.png")
        title = f"{args.dataset.upper()}: {prob_type.replace('_', ' ').title()} by Review Bucket"
        plot_bucket_scatter(
            points, prob_type, output_path, title,
        )
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
