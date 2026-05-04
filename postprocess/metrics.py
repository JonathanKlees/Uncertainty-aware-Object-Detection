"""
Core divergence metrics operating on probability distributions.

All metrics here operate on Dict[str, float] probability distributions.
For legacy functions that operate on counts with built-in smoothing, see metrics_legacy.py.
For dataset comparison and bootstrap utilities, see metrics_comparison.py.
"""

import math
from enum import Enum
from typing import Dict

from postprocess.smoothing import Distribution


class Metric(Enum):
    """Divergence metrics (all operate on probability distributions)."""
    KL = "kl"
    JS = "js"
    JS_SQRT = "js_sqrt"


def kl_divergence_probs(P: Distribution, Q: Distribution) -> float:
    """
    KL(P || Q) for already-normalized probability distributions.
    
    Args:
        P: Reference distribution (true distribution)
        Q: Candidate distribution (predicted distribution)
    
    Returns:
        KL divergence value
    
    Note: Returns inf if Q[k] == 0 where P[k] > 0.
    """
    kl = 0.0
    for k, p in P.items():
        if p > 0.0:
            q = Q.get(k, 0.0)
            if q <= 0.0:
                return math.inf
            kl += p * math.log(p / q)
    return kl





def js_divergence_from_probs(
    P: Distribution,
    Q: Distribution,
    sqrt: bool = True,
) -> float:
    """
    Jensen-Shannon divergence on probability distributions.
    
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = (P + Q) / 2
    
    Args:
        P: First probability distribution
        Q: Second probability distribution  
        sqrt: If True, return sqrt(JS) which is a proper metric
    
    Returns:
        JS divergence (or sqrt thereof)
    
    Note: May return inf/nan if distributions contain zeros where the other doesn't.
    """
    support = set(P.keys()) | set(Q.keys())
    
    # Ensure both have full support
    P_full = {k: P.get(k, 0.0) for k in support}
    Q_full = {k: Q.get(k, 0.0) for k in support}
    
    M = {k: 0.5 * (P_full[k] + Q_full[k]) for k in support}
    
    js = 0.5 * kl_divergence_probs(P_full, M) + 0.5 * kl_divergence_probs(Q_full, M)
    # Handle floating point errors causing small negative values
    js = max(0.0, js)
    return math.sqrt(js) if sqrt else js


def compute_metric(
    P: Distribution,
    Q: Distribution,
    metric: Metric,
) -> float:
    """
    Compute divergence metric between two probability distributions.
    
    Args:
        P: Reference distribution (e.g., unbiased)
        Q: Candidate distribution (e.g., postprocessed)
        metric: Which metric to compute
    
    Returns:
        Divergence value
    """
    if metric == Metric.KL:
        return kl_divergence_probs(P, Q)
    elif metric == Metric.JS:
        return js_divergence_from_probs(P, Q, sqrt=False)
    elif metric == Metric.JS_SQRT:
        return js_divergence_from_probs(P, Q, sqrt=True)
    else:
        raise ValueError(f"Unknown metric: {metric}")
