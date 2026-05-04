"""
Smoothing utilities for probability distributions.

Smoothing is part of the post-processing method pipeline, not the metric computation.
"""

from enum import Enum
from typing import Dict, List


# Type aliases
Counts = Dict[str, int]
Dataset = Dict[str, Counts]
Distribution = Dict[str, float]
DistDataset = Dict[str, Distribution]


class Smoothing(Enum):
    """Smoothing options for post-processing pipeline."""
    DIRICHLET_POST = "dirichlet_post"  # counts → PP → counts → Dirichlet(α) → probs
    DIRICHLET_PRE = "dirichlet_pre"    # counts → Dirichlet(α) → PP → probs
    EPSILON = "epsilon"                 # counts → PP → probs → ε-smooth


def counts_to_dirichlet_probs(
    counts: Counts,
    *,
    alpha: float,
    support: List[str],
) -> Distribution:
    """
    Convert raw counts into probabilities using a symmetric Dirichlet prior.

    p_i = (n_i + alpha) / (N + alpha * K)
    
    Args:
        counts: Raw count dictionary
        alpha: Dirichlet smoothing parameter (typically 0.5 or 1.0)
        support: List of class names (defines the support)
    
    Returns:
        Smoothed probability distribution
    """
    K = len(support)
    N = sum(counts.get(k, 0) for k in support)
    denom = N + alpha * K
    return {k: (counts.get(k, 0) + alpha) / denom for k in support}


def mle_probs(counts: Counts, support: set[str] = None) -> Distribution:
    """
    Maximum likelihood estimate of probabilities from counts.
    
    Args:
        counts: Raw count dictionary
        support: Set of class names; if None, inferred from counts keys
    
    Returns:
        MLE probability distribution (may contain zeros)
    """
    if support is None:
        support = set(counts.keys())

    N = sum(counts.get(k, 0) for k in support)
    if N == 0:
        return {k: 0.0 for k in support}
    return {k: counts.get(k, 0) / N for k in support}


def epsilon_smooth_probs(
    probs: Distribution,
    epsilon: float = 1e-10,
) -> Distribution:
    """
    Add epsilon smoothing to probabilities to avoid zeros.
    
    Args:
        probs: Probability distribution (may contain zeros)
        epsilon: Small value to add to each probability
    
    Returns:
        Smoothed and renormalized probability distribution
    """
    smoothed = {k: v + epsilon for k, v in probs.items()}
    total = sum(smoothed.values())
    return {k: v / total for k, v in smoothed.items()}


def probs_to_counts(probs: Distribution, total: int, support: List[str]) -> Counts:
    """
    Convert probabilities to integer counts preserving total exactly.
    
    Uses largest-remainder rounding to distribute the total across classes.
    
    Args:
        probs: Probability distribution (will be normalized if not summing to 1)
        total: Target total count
        support: List of class names (defines output keys)
    
    Returns:
        Count dictionary with all support keys (including zeros)
    """
    import math
    
    if total <= 0:
        return {c: 0 for c in support}
    
    # Ensure non-negative and get sum
    items = [(c, max(0.0, float(probs.get(c, 0.0)))) for c in support]
    s = sum(v for _, v in items)
    if s <= 0:
        return {c: 0 for c in support}
    
    # Scale to total and floor
    scaled = [(c, v / s * total) for c, v in items]
    floored = {c: int(math.floor(x)) for c, x in scaled}
    remainder = total - sum(floored.values())
    
    # Distribute remainder by largest fractional parts
    if remainder > 0:
        fracs = sorted(scaled, key=lambda kv: (kv[1] - math.floor(kv[1])), reverse=True)
        for c, _ in fracs[:remainder]:
            floored[c] += 1
    
    return floored


def smooth_counts(
    counts: Counts,
    smoothing: Smoothing,
    classes: List[str],
    alpha: float = 0.5,
    epsilon: float = 1e-10,
) -> Distribution:
    """
    Apply smoothing to counts, returning a probability distribution.
    
    Args:
        counts: Raw count dictionary
        smoothing: Type of smoothing to apply
        classes: List of class names (defines support)
        alpha: Dirichlet smoothing parameter
        epsilon: Epsilon smoothing parameter
    
    Returns:
        Smoothed probability distribution
    """
    if smoothing == Smoothing.DIRICHLET_POST or smoothing == Smoothing.DIRICHLET_PRE:
        return counts_to_dirichlet_probs(counts, alpha=alpha, support=classes)
    elif smoothing == Smoothing.EPSILON:
        probs = mle_probs(counts, set(classes))
        return epsilon_smooth_probs(probs, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown smoothing: {smoothing}")
