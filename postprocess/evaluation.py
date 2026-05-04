"""
Evaluation utilities for comparing distributions and measuring post-processing impact.

For legacy counts-based evaluation, see evaluation_legacy.py.
"""

from typing import Dict, List, Optional, Tuple

from postprocess.smoothing import (
    Dataset,
    DistDataset,
    Smoothing,
    smooth_counts,
)
from postprocess.metrics import Metric
from postprocess.metrics_comparison import (
    DivergenceRecord,
    compute_all_divergences,
    summarize_records,
    extract_deltas,
    bootstrap_delta_summary,
)


# =============================================================================
# Utility functions
# =============================================================================

def majority_label(d: Dict[str, float]) -> str:
    """Return the majority label from a distribution, or 'undecided' if tied/empty."""
    if not d:
        return "undecided"
    max_val = max(d.values())
    max_keys = [k for k, v in d.items() if v == max_val]
    return max_keys[0] if len(max_keys) == 1 else "undecided"


def count_majority_changes(
    map_id_dist_a: Dict[str, Dict[str, float]],
    map_id_dist_b: Dict[str, Dict[str, float]],
) -> Tuple[int, int]:
    """
    Count how many objects have different majority labels between two distributions.
    
    Returns:
        (changes, total) tuple
    """
    changes = 0
    total = 0
    
    for obj_id in map_id_dist_a:
        dist_a = map_id_dist_a.get(obj_id)
        dist_b = map_id_dist_b.get(obj_id)
        
        if dist_a is None or dist_b is None:
            continue
        
        majority_a = majority_label(dist_a)
        majority_b = majority_label(dist_b)
        
        if majority_a != majority_b:
            changes += 1
        total += 1
    
    return changes, total


def print_majority_changes(
    map_id_dist_a: Dict[str, Dict[str, float]],
    map_id_dist_b: Dict[str, Dict[str, float]],
) -> None:
    """Count and print majority label changes between two distributions."""
    changes, total = count_majority_changes(map_id_dist_a, map_id_dist_b)
    pct = (changes / total * 100.0) if total else 0.0
    print(f"Majority membership changes (proposed class): {changes}/{total} ({pct:.2f}%)")


# =============================================================================
# Printing utilities
# =============================================================================

def print_evaluation_results(
    name: str, 
    results: Dict, 
    baseline_js_sqrt: float,
) -> None:
    """
    Print evaluation results for a post-processing method.
    
    Supports both new format (all_metrics) and legacy format (distribution_comparison).
    
    Args:
        name: Method name
        results: Evaluation results dict (from evaluate_method or evaluate_postprocess_method)
        baseline_js_sqrt: Baseline JS_SQRT mean for percentage calculations
    """
    print(f"\n=== Post-process method: {name} ===")
    
    avg = results["avg_prob_proposed"]
    bias_diff = abs(avg['biased'] - avg['unbiased'])
    pct_reduction = avg['reduction'] * 100 / bias_diff if bias_diff > 0 else float('nan')
    print(f"Avg P(proposed): unbiased={avg['unbiased']:.4f}, "
          f"biased={avg['biased']:.4f}, postproc={avg['postprocessed']:.4f} -> "
          f"reduction={avg['reduction']:.4f} ({pct_reduction:+.2f}%)")
    
    # Non-bootstrapped metrics overview
    print("Distribution comparison (non-bootstrapped) postprocessed:")
    if "all_metrics" in results:
        # New format: all_metrics has kl, js, js_sqrt
        all_metrics = results["all_metrics"]
        for metric_name in ["kl", "js", "js_sqrt"]:
            stats = all_metrics[metric_name]
            print(f"  {metric_name}: {stats['mean']:.6f}+-{stats['std']:.6f}, [{stats['median']:.6f}]")
    else:
        # Legacy format: distribution_comparison is dict of metric -> stats
        for metric_name, stats in results["distribution_comparison"].items():
            mean = stats.get("mean", float("nan"))
            median = stats.get("median", float("nan"))
            std = stats.get("std", float("nan"))
            print(f"  {metric_name}: {mean:.6f}+-{std:.6f}, [{median:.6f}]")
    
    # Bootstrapped delta summary
    delta = results["delta_summary"]
    metric_name = results.get("metric", "JS_SQRT")
    print(f"Post-processing impact on divergence ({metric_name}, Δ = after - before) [bootstrapped]")
    print("────────────────────────────────────────────────────────")
    pct_of_baseline = delta['mean']['point'] * 100 / baseline_js_sqrt if baseline_js_sqrt > 0 else float('nan')
    print(
        f"Mean Δ-divergence:   {delta['mean']['point']:+.5f} "
        f"(95% CI [{delta['mean']['ci'][0]:+.5f}, "
        f"{delta['mean']['ci'][1]:+.5f}]) -> ({pct_of_baseline:+.2f}%)"
    )
    print(
        f"Median Δ-divergence: {delta['median']['point']:+.5f} "
        f"(95% CI [{delta['median']['ci'][0]:+.5f}, "
        f"{delta['median']['ci'][1]:+.5f}])"
    )
    print(f"Fraction improved:  {delta['fraction_improved']:.1%}")
    print(f"Fraction worsened:  {delta['fraction_worsened']:.1%}")


def print_baseline_comparison(
    map_id_unbiased_counts: Dataset,
    map_id_biased_counts: Dataset,
    classes: List[str],
    alpha: float = 0.5,
    smoothing: Smoothing = Smoothing.DIRICHLET_POST,
    epsilon: float = 1e-10,
) -> Dict:
    """
    Print comparison between unbiased and biased distributions.
    
    Args:
        map_id_unbiased_counts: Ground truth counts
        map_id_biased_counts: Biased counts
        classes: List of class names
        alpha: Dirichlet smoothing parameter
        smoothing: Smoothing type (default: DIRICHLET_POST)
        epsilon: Epsilon smoothing parameter (for EPSILON smoothing)
    
    Returns:
        Dict with summary statistics (keys: kl, js, js_sqrt)
    """
    # Convert counts to probability distributions
    unbiased_probs = {
        k: smooth_counts(c, smoothing, classes, alpha, epsilon)
        for k, c in map_id_unbiased_counts.items()
    }
    biased_probs = {
        k: smooth_counts(c, smoothing, classes, alpha, epsilon)
        for k, c in map_id_biased_counts.items()
    }
    
    # Compute divergences using unified approach (pass biased as both biased and postproc)
    records = compute_all_divergences(unbiased_probs, biased_probs, biased_probs)
    summary = summarize_records(records, "biased")
    
    print("\nDistribution comparison (non-bootstrapped):")
    for metric, stats in summary.items():
        mean = stats.get("mean", float("nan"))
        median = stats.get("median", float("nan"))
        std = stats.get("std", float("nan"))
        print(f"  {metric}: {mean:.6f}+-{std:.6f}, [{median:.6f}]")

    return summary


# =============================================================================
# Evaluation on Probability Distributions
# =============================================================================

def evaluate_method(
    unbiased_probs: DistDataset,
    biased_probs: DistDataset,
    postprocessed_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
    metric: Metric = Metric.JS_SQRT,
    sample_plot_path: Optional[str] = None,
    num_samples: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Evaluate a post-processing method on probability distributions.
    
    Smoothing is part of the method, not evaluation. All inputs should be
    probability distributions (method already applied smoothing).
    
    Args:
        unbiased_probs: Ground truth probability distributions
        biased_probs: Biased probability distributions (before post-processing)
        postprocessed_probs: Post-processed probability distributions
        map_id_proposed_class: Proposed class per object
        metric: Divergence metric for delta computation (default: JS_SQRT)
        sample_plot_path: Optional path to save sample distribution plots
        num_samples: Number of samples per category for plots (default: 10)
        seed: Random seed for sample selection (default: 42)
    
    Returns:
        Dict with evaluation metrics including:
        - avg_prob_proposed: Average probability of proposed class
        - all_metrics: Non-bootstrapped KL, JS, JS_SQRT summaries
        - delta_summary: Bootstrapped delta statistics
        - divergence_records: Pre-computed divergence records (if needed elsewhere)
    """
    # Calculate average probability of proposed class
    def avg_prop(map_id_probs: DistDataset) -> float:
        probs = []
        for obj_id, proposed_cls in map_id_proposed_class.items():
            dist = map_id_probs.get(obj_id)
            if dist:
                probs.append(dist.get(proposed_cls, 0.0))
        return sum(probs) / len(probs) if probs else 0.0
    
    avg_unbiased = avg_prop(unbiased_probs)
    avg_biased = avg_prop(biased_probs)
    avg_postproc = avg_prop(postprocessed_probs)
    
    # Compute all divergences in a single pass
    records = compute_all_divergences(unbiased_probs, biased_probs, postprocessed_probs)
    
    # Extract statistics from records
    all_metrics_postprocessed = summarize_records(records, "postproc")
    all_metrics_biased = summarize_records(records, "biased")
    
    # Bootstrapped deltas using specified metric
    deltas = extract_deltas(records, metric)
    delta_summary = bootstrap_delta_summary(deltas)
    
    # Generate sample distribution plots if requested
    if sample_plot_path:
        from postprocess.visualization import plot_sample_distributions
        plot_sample_distributions(
            records=records,
            unbiased_probs=unbiased_probs,
            biased_probs=biased_probs,
            postprocessed_probs=postprocessed_probs,
            map_id_proposed_class=map_id_proposed_class,
            output_path=sample_plot_path,
            num_samples=num_samples,
            metric=metric,
            seed=seed,
        )
    
    return {
        "avg_prob_proposed": {
            "unbiased": avg_unbiased,
            "biased": avg_biased,
            "postprocessed": avg_postproc,
            "reduction": abs(avg_postproc - avg_unbiased) - abs(avg_biased - avg_unbiased),
        },
        "all_metrics": all_metrics_postprocessed,
        "all_metrics_biased": all_metrics_biased,
        "delta_summary": delta_summary,
        "metric": metric.name,
        "divergence_records": records,
    }


# =============================================================================
# Annotation Timing & Damper Analysis
# =============================================================================

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class SpeedupMetrics:
    """Annotation time statistics for biased vs unbiased conditions."""
    biased_mean: float
    biased_median: float
    biased_std: float
    biased_count: int
    unbiased_mean: float
    unbiased_median: float
    unbiased_std: float
    unbiased_count: int
    speedup_mean: float  # unbiased_mean / biased_mean
    speedup_median: float  # unbiased_median / biased_median


def compute_speedup_metrics(
    timing_data: Dict[str, Any],  # Dict[str, AnnotationTiming]
) -> SpeedupMetrics:
    """
    Compute annotation time speedup metrics.
    
    Aggregates all biased and unbiased annotation times across all objects
    to compute mean, median, std and speedup ratios.
    
    Args:
        timing_data: Dict mapping object_id -> AnnotationTiming
        
    Returns:
        SpeedupMetrics with aggregated statistics
    """
    all_biased_times = []
    all_unbiased_times = []
    
    for timing in timing_data.values():
        all_biased_times.extend(timing.biased_times)
        all_unbiased_times.extend(timing.unbiased_times)
    
    if not all_biased_times:
        biased_mean, biased_median, biased_std = 0.0, 0.0, 0.0
    else:
        biased_mean = float(np.mean(all_biased_times))
        biased_median = float(np.median(all_biased_times))
        biased_std = float(np.std(all_biased_times))
    
    if not all_unbiased_times:
        unbiased_mean, unbiased_median, unbiased_std = 0.0, 0.0, 0.0
    else:
        unbiased_mean = float(np.mean(all_unbiased_times))
        unbiased_median = float(np.median(all_unbiased_times))
        unbiased_std = float(np.std(all_unbiased_times))
    
    # Speedup ratio (how much faster biased is compared to unbiased)
    speedup_mean = unbiased_mean / biased_mean if biased_mean > 0 else float('inf')
    speedup_median = unbiased_median / biased_median if biased_median > 0 else float('inf')
    
    return SpeedupMetrics(
        biased_mean=biased_mean,
        biased_median=biased_median,
        biased_std=biased_std,
        biased_count=len(all_biased_times),
        unbiased_mean=unbiased_mean,
        unbiased_median=unbiased_median,
        unbiased_std=unbiased_std,
        unbiased_count=len(all_unbiased_times),
        speedup_mean=speedup_mean,
        speedup_median=speedup_median,
    )


@dataclass
class DamperResult:
    """Damper analysis results with per-class breakdown."""
    global_dampers: List[float]  # All valid damper values
    global_mean: float
    global_median: float
    per_class_dampers: Dict[str, List[float]]  # class -> list of dampers
    per_class_stats: Dict[str, Dict[str, float]]  # class -> {mean, median, count}
    infinite_count: int  # Objects where unbiased_prob[proposed] = 0


def compute_damper_values(
    biased_probs: DistDataset,
    unbiased_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
) -> DamperResult:
    """
    Compute damper values showing how to correct biased probabilities.
    
    Damper = unbiased_prob[proposed] / biased_prob[proposed]
    Such that: biased * damper = unbiased
    
    - Values < 1: proposal inflated probability (need to scale down)
    - Values > 1: proposal deflated probability (need to scale up)
    - Values = 1: no effect
    
    Args:
        biased_probs: Biased probability distributions
        unbiased_probs: Unbiased (ground truth) probability distributions
        map_id_proposed_class: Proposed class per object
        
    Returns:
        DamperResult with global and per-class damper statistics
    """
    global_dampers: List[float] = []
    per_class_dampers: Dict[str, List[float]] = {}
    infinite_count = 0
    
    for obj_id, proposed_cls in map_id_proposed_class.items():
        biased = biased_probs.get(obj_id)
        unbiased = unbiased_probs.get(obj_id)
        
        if biased is None or unbiased is None:
            continue
        
        biased_prob = biased.get(proposed_cls, 0.0)
        unbiased_prob = unbiased.get(proposed_cls, 0.0)
        
        # Handle division by zero
        if biased_prob <= 1e-10:
            infinite_count += 1
            continue
        
        damper = unbiased_prob / biased_prob
        global_dampers.append(damper)
        
        # Per-class breakdown
        if proposed_cls not in per_class_dampers:
            per_class_dampers[proposed_cls] = []
        per_class_dampers[proposed_cls].append(damper)
    
    # Compute global stats
    global_mean = float(np.mean(global_dampers)) if global_dampers else 0.0
    global_median = float(np.median(global_dampers)) if global_dampers else 0.0
    
    # Compute per-class stats
    per_class_stats: Dict[str, Dict[str, float]] = {}
    for cls, dampers in per_class_dampers.items():
        per_class_stats[cls] = {
            "mean": float(np.mean(dampers)),
            "median": float(np.median(dampers)),
            "count": len(dampers),
        }
    
    return DamperResult(
        global_dampers=global_dampers,
        global_mean=global_mean,
        global_median=global_median,
        per_class_dampers=per_class_dampers,
        per_class_stats=per_class_stats,
        infinite_count=infinite_count,
    )


def compute_per_class_dampers(
    biased_probs: DistDataset,
    unbiased_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    use_median: bool = True,
) -> Dict[str, float]:
    """
    Compute per-class damper weights from training data.
    
    Returns a dict mapping class name -> damper weight (median or mean).
    For classes with no training samples, returns 1.0 (no change).
    
    Formula: damper = unbiased_prob[proposed] / biased_prob[proposed]
    Such that: biased * damper = unbiased
    
    Args:
        biased_probs: Biased probability distributions (training)
        unbiased_probs: Unbiased (ground truth) probability distributions (training)
        map_id_proposed_class: Proposed class per object
        classes: List of all class names
        use_median: If True, use median (robust), else mean
        
    Returns:
        Dict[str, float] mapping class name -> damper weight
    """
    # First compute the full damper result to get per-class damper lists
    result = compute_damper_values(biased_probs, unbiased_probs, map_id_proposed_class)
    
    # Extract median/mean per class, default to 1.0 for unseen classes
    per_class_weights: Dict[str, float] = {}
    for cls in classes:
        if cls in result.per_class_stats:
            stats = result.per_class_stats[cls]
            per_class_weights[cls] = stats["median"] if use_median else stats["mean"]
        else:
            per_class_weights[cls] = 1.0  # No change for unseen classes
    
    return per_class_weights


def print_speedup_metrics(metrics: SpeedupMetrics) -> None:
    """Print annotation speedup metrics."""
    print("\n" + "=" * 60)
    print("ANNOTATION TIME ANALYSIS")
    print("=" * 60)
    print(f"\nBiased (with proposal shown):")
    print(f"  Mean time:   {metrics.biased_mean:.3f}s ± {metrics.biased_std:.3f}s")
    print(f"  Median time: {metrics.biased_median:.3f}s")
    print(f"  Count:       {metrics.biased_count} annotations")
    
    print(f"\nUnbiased (no proposal shown):")
    print(f"  Mean time:   {metrics.unbiased_mean:.3f}s ± {metrics.unbiased_std:.3f}s")
    print(f"  Median time: {metrics.unbiased_median:.3f}s")
    print(f"  Count:       {metrics.unbiased_count} annotations")
    
    print(f"\nSpeedup (unbiased / biased):")
    print(f"  Mean speedup:   {metrics.speedup_mean:.2f}x")
    print(f"  Median speedup: {metrics.speedup_median:.2f}x")


def print_damper_results(result: DamperResult) -> None:
    """Print damper analysis results with per-class breakdown."""
    print("\n" + "=" * 60)
    print("DAMPER ANALYSIS (unbiased_prob / biased_prob)")
    print("  biased * damper = unbiased")
    print("  damper < 1 means proposal inflated probability")
    print("=" * 60)
    print(f"\nGlobal statistics:")
    print(f"  Mean damper:   {result.global_mean:.3f}")
    print(f"  Median damper: {result.global_median:.3f}")
    print(f"  Valid objects: {len(result.global_dampers)}")
    print(f"  Excluded (unbiased=0): {result.infinite_count}")
    
    print(f"\nPer-class breakdown:")
    print(f"  {'Class':<15} {'Mean':>8} {'Median':>8} {'Count':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    
    # Sort by count descending
    sorted_classes = sorted(
        result.per_class_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    for cls, stats in sorted_classes:
        print(f"  {cls:<15} {stats['mean']:>8.3f} {stats['median']:>8.3f} {stats['count']:>8}")
