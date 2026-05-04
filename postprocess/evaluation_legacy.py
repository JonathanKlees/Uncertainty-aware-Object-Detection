"""
Legacy evaluation utilities for counts-based post-processing methods.

These functions work with Dataset (counts) rather than DistDataset (probs).
For new code, use evaluate_method() from evaluation.py with probability distributions.
"""

from typing import Dict, List, Optional, Union

from postprocess.smoothing import (
    Dataset,
    DistDataset,
    Smoothing,
    smooth_counts,
)
from postprocess.metrics import Metric
from postprocess.metrics_comparison import (
    bootstrap_delta_summary,
)
from postprocess.metrics_legacy import (
    compare_datasets,
    divergence_deltas_legacy,
)


# =============================================================================
# Helper functions (counts-based)
# =============================================================================

def prob_of_class(counts: Dict[str, int], cls: str) -> float:
    """Compute probability of a class from counts."""
    if not counts:
        return 0.0
    total = sum(counts.values())
    return counts.get(cls, 0) / total if total > 0 else 0.0


def avg_prob_of_proposed(
    map_id_counts: Dataset, 
    map_id_proposed_class: Dict[str, str]
) -> float:
    """Compute average probability of proposed class across all objects (counts-based)."""
    probs = []
    for obj_id, proposed_cls in map_id_proposed_class.items():
        counts = map_id_counts.get(obj_id)
        if counts:
            probs.append(prob_of_class(counts, proposed_cls))
    return sum(probs) / len(probs) if probs else 0.0


# =============================================================================
# Legacy Evaluation (counts-based)
# =============================================================================

def evaluate_postprocess_method_legacy(
    map_id_unbiased_counts: Dataset,
    map_id_biased_counts: Dataset,
    map_id_postprocessed: Dataset,
    map_id_proposed_class: Dict[str, str],
    alpha: float = 0.5,
) -> Dict:
    """
    Evaluate a post-processing method against unbiased ground truth.
    
    LEGACY: This function works with counts and uses legacy delta computation.
    For new code, use evaluate_method() from evaluation.py with probability distributions.
    
    Returns dict with evaluation metrics.
    """
    # Calculate average probability of proposed class
    avg_unbiased = avg_prob_of_proposed(map_id_unbiased_counts, map_id_proposed_class)
    avg_biased = avg_prob_of_proposed(map_id_biased_counts, map_id_proposed_class)
    avg_postproc = avg_prob_of_proposed(map_id_postprocessed, map_id_proposed_class)
    
    # Compare distributions
    summary = compare_datasets(map_id_unbiased_counts, map_id_postprocessed, alpha=alpha)
    
    # Calculate delta improvements (legacy uses dirichlet JS divergence)
    deltas = divergence_deltas_legacy(map_id_unbiased_counts, map_id_biased_counts, map_id_postprocessed, alpha=alpha)
    delta_summary = bootstrap_delta_summary(deltas)
    
    return {
        "avg_prob_proposed": {
            "unbiased": avg_unbiased,
            "biased": avg_biased,
            "postprocessed": avg_postproc,
            "reduction": abs(avg_postproc - avg_unbiased) - abs(avg_biased - avg_unbiased),
        },
        "distribution_comparison": summary,
        "delta_summary": delta_summary,
    }

