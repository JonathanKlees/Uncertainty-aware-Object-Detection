"""Dataset comparison utilities for probability distributions.

All functions work on DistDataset (probability distributions).
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from postprocess.smoothing import (
    Counts,
    Dataset,
    Distribution,
    DistDataset,
    Smoothing,
    smooth_counts,
    epsilon_smooth_probs,
)
from postprocess.metrics import Metric, compute_metric


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class DivergenceRecord:
    """
    Pre-computed divergences for a single object.
    
    Stores divergences for both biased and post-processed distributions
    compared to ground truth (unbiased).
    """
    biased_kl: float
    biased_js: float
    biased_js_sqrt: float
    postproc_kl: float
    postproc_js: float
    postproc_js_sqrt: float
    
    def delta(self, metric: Metric) -> float:
        """Compute delta = postproc - biased for the given metric."""
        if metric == Metric.KL:
            return self.postproc_kl - self.biased_kl
        elif metric == Metric.JS:
            return self.postproc_js - self.biased_js
        else:  # JS_SQRT
            return self.postproc_js_sqrt - self.biased_js_sqrt
    
    def biased(self, metric: Metric) -> float:
        """Get biased divergence for the given metric."""
        if metric == Metric.KL:
            return self.biased_kl
        elif metric == Metric.JS:
            return self.biased_js
        else:
            return self.biased_js_sqrt
    
    def postproc(self, metric: Metric) -> float:
        """Get post-processed divergence for the given metric."""
        if metric == Metric.KL:
            return self.postproc_kl
        elif metric == Metric.JS:
            return self.postproc_js
        else:
            return self.postproc_js_sqrt


# =============================================================================
# Helper functions
# =============================================================================

def _mean(xs: List[float]) -> float:
    """Mean of a list of floats."""
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    """Population standard deviation."""
    if not xs:
        return float("nan")
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def _percentile(xs: List[float], q: float) -> float:
    """
    Simple percentile with linear interpolation.
    q in [0,1]
    """
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


# =============================================================================
# Unified divergence computation
# =============================================================================

def compute_all_divergences(
    unbiased_probs: DistDataset,
    biased_probs: DistDataset,
    postprocessed_probs: DistDataset,
) -> Dict[str, DivergenceRecord]:
    """
    Compute all divergences in a single pass.
    
    For each object, computes KL, JS, JS_SQRT divergences for both:
    - biased vs unbiased (ground truth)
    - post-processed vs unbiased (ground truth)
    
    This avoids redundant computation when multiple metrics are needed.
    
    Args:
        unbiased_probs: Ground truth probability distributions
        biased_probs: Biased probability distributions (before post-processing)
        postprocessed_probs: Post-processed probability distributions
    
    Returns:
        Dict mapping object_id to DivergenceRecord
    """
    records = {}
    
    # Only process objects that exist in all three datasets
    common_keys = unbiased_probs.keys() & biased_probs.keys() & postprocessed_probs.keys()
    
    for obj_id in common_keys:
        unbiased = unbiased_probs[obj_id]
        biased = biased_probs[obj_id]
        postproc = postprocessed_probs[obj_id]
        
        records[obj_id] = DivergenceRecord(
            biased_kl=compute_metric(unbiased, biased, Metric.KL),
            biased_js=compute_metric(unbiased, biased, Metric.JS),
            biased_js_sqrt=compute_metric(unbiased, biased, Metric.JS_SQRT),
            postproc_kl=compute_metric(unbiased, postproc, Metric.KL),
            postproc_js=compute_metric(unbiased, postproc, Metric.JS),
            postproc_js_sqrt=compute_metric(unbiased, postproc, Metric.JS_SQRT),
        )
    
    return records


def summarize_records(
    records: Dict[str, DivergenceRecord],
    source: str,  # "biased" or "postproc"
) -> Dict[str, Dict[str, float]]:
    """
    Extract summary statistics from divergence records.
    
    Args:
        records: Pre-computed divergence records
        source: Which divergences to summarize: "biased" or "postproc"
    
    Returns:
        {"kl": {mean, median, std}, "js": {...}, "js_sqrt": {...}}
    """
    if source == "biased":
        kl_vals = [r.biased_kl for r in records.values()]
        js_vals = [r.biased_js for r in records.values()]
        js_sqrt_vals = [r.biased_js_sqrt for r in records.values()]
    else:  # postproc
        kl_vals = [r.postproc_kl for r in records.values()]
        js_vals = [r.postproc_js for r in records.values()]
        js_sqrt_vals = [r.postproc_js_sqrt for r in records.values()]
    
    return {
        "kl": {"mean": _mean(kl_vals), "median": _percentile(kl_vals, 0.5), "std": _std(kl_vals)},
        "js": {"mean": _mean(js_vals), "median": _percentile(js_vals, 0.5), "std": _std(js_vals)},
        "js_sqrt": {"mean": _mean(js_sqrt_vals), "median": _percentile(js_sqrt_vals, 0.5), "std": _std(js_sqrt_vals)},
    }


def extract_deltas(
    records: Dict[str, DivergenceRecord],
    metric: Metric = Metric.JS_SQRT,
) -> Dict[str, float]:
    """
    Extract deltas from divergence records for bootstrapping.
    
    Args:
        records: Pre-computed divergence records
        metric: Which metric's delta to extract
    
    Returns:
        Dict mapping object_id to delta (negative = improved)
    """
    return {obj_id: rec.delta(metric) for obj_id, rec in records.items()}


# =============================================================================
# Bootstrapping
# =============================================================================

def bootstrap_delta_summary(
    deltas: Dict[str, float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict:
    """Bootstrap summary statistics for divergence deltas."""
    rng = random.Random(seed)
    keys = list(deltas.keys())
    n = len(keys)

    boot_means = []
    boot_medians = []

    for _ in range(n_boot):
        sample = [deltas[rng.choice(keys)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
        boot_medians.append(sorted(sample)[n // 2])

    boot_means.sort()
    boot_medians.sort()

    l_deltas = list(deltas.values())

    return {
        "mean": {
            "point": _mean(l_deltas),
            "ci": (_percentile(boot_means, ci), _percentile(boot_means, 1.0 - ci)),
        },
        "median": {
            "point": _percentile(l_deltas, 0.5),
            "ci": (_percentile(boot_medians, ci), _percentile(boot_medians, 1.0 - ci)),
        },
        "fraction_improved": sum(d < 0 for d in deltas.values()) / n,
        "fraction_worsened": sum(d > 0 for d in deltas.values()) / n,
    }

