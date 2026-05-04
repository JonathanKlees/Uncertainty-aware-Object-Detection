"""
Legacy metrics that operate on counts with built-in smoothing.

These functions combine smoothing + metric computation in one call.
For the new modular approach, use smoothing.py + metrics.py separately.
"""

import math
import random
from typing import Dict, List, Literal, Optional

from postprocess.smoothing import (
    Counts,
    Distribution,
    counts_to_dirichlet_probs,
    mle_probs,
)
from postprocess.metrics import kl_divergence_probs


Mode = Literal["keys", "within", "both"]


def _support_union(*ds: Counts) -> set[str]:
    """Get union of keys across multiple count dictionaries."""
    s: set[str] = set()
    for d in ds:
        s.update(d.keys())
    return s


def kl_divergence_native_from_counts(
    P_counts: Counts,
    Q_counts: Counts,
) -> float:
    """
    UNSAFE: native KL computed from raw MLE probabilities.
    
    Included only for comparison / diagnostics.
    May return inf or crash if Q has zero counts where P doesn't.
    """
    support = _support_union(P_counts, Q_counts)
    P = mle_probs(P_counts, support)
    Q = mle_probs(Q_counts, support)
    return kl_divergence_probs(P, Q)

def kl_divergence_epsilon(
    P: Distribution,
    Q: Distribution,
    epsilon: float = 1e-10,
) -> float:
    """
    KL(P || Q) with internal epsilon smoothing to avoid division by zero.
    
    Args:
        P: Reference distribution
        Q: Candidate distribution
        epsilon: Small value added to both distributions
    
    Returns:
        KL divergence value (always finite)
    """
    kl_div = 0.0
    for key in P:
        p = P[key] + epsilon
        q = Q.get(key, 0) + epsilon
        kl_div += p * math.log(p / q)
    return kl_div


def kl_divergence_epsilon_from_counts(
    P_counts: Counts,
    Q_counts: Counts,
) -> float:
    """
    KL divergence from counts using epsilon smoothing internally.
    
    Numerically stable due to epsilon, but not principled smoothing.
    """
    support = _support_union(P_counts, Q_counts)
    P = mle_probs(P_counts, support)
    Q = mle_probs(Q_counts, support)
    return kl_divergence_epsilon(P, Q)


def kl_divergence_dirichlet_from_counts(
    P_counts: Counts,
    Q_counts: Counts,
    *,
    alpha: float = 1.0,
) -> float:
    """
    KL(P || Q) after Dirichlet smoothing.
    
    Still asymmetric, but numerically stable.
    """
    support = list(_support_union(P_counts, Q_counts))
    P = counts_to_dirichlet_probs(P_counts, alpha=alpha, support=support)
    Q = counts_to_dirichlet_probs(Q_counts, alpha=alpha, support=support)
    return kl_divergence_probs(P, Q)


def js_divergence_dirichlet_from_counts(
    P_counts: Counts,
    Q_counts: Counts,
    *,
    alpha: float = 1.0,
    sqrt: bool = True,
) -> float:
    """
    Jensen-Shannon divergence on raw counts using Dirichlet smoothing.

    This is the recommended legacy metric.
    
    Args:
        P_counts: First count distribution
        Q_counts: Second count distribution
        alpha: Dirichlet smoothing parameter
        sqrt: If True, return sqrt(JS) (a proper metric)
    
    Returns:
        JS divergence (or sqrt thereof)
    """
    support = list(_support_union(P_counts, Q_counts))
    P = counts_to_dirichlet_probs(P_counts, alpha=alpha, support=support)
    Q = counts_to_dirichlet_probs(Q_counts, alpha=alpha, support=support)

    M = {k: 0.5 * (P[k] + Q[k]) for k in support}

    js = 0.5 * kl_divergence_probs(P, M) + 0.5 * kl_divergence_probs(Q, M)
    return math.sqrt(js) if sqrt else js


# =============================================================================
# Legacy delta utilities (use metrics_comparison.py for modular versions)
# =============================================================================

Dataset = Dict[str, Counts]


def per_key_divergence_legacy(
    reference: Dataset,
    candidate: Dataset,
    *,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Compute JS divergence for each key (legacy, uses dirichlet+js_sqrt)."""
    out = {}
    for k in reference.keys() & candidate.keys():
        out[k] = js_divergence_dirichlet_from_counts(
            reference[k],
            candidate[k],
            alpha=alpha,
            sqrt=True,
        )
    return out


def divergence_deltas_legacy(
    reference: Dataset,
    before: Dataset,
    after: Dataset,
    *,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Compute delta = after - before for each key's divergence (legacy)."""
    d_before = per_key_divergence_legacy(reference, before, alpha=alpha)
    d_after = per_key_divergence_legacy(reference, after, alpha=alpha)

    return {
        k: d_after[k] - d_before[k]
        for k in d_before.keys() & d_after.keys()
    }


# =============================================================================
# Legacy comparison functions (moved from metrics_comparison.py)
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


def _resample_counts_from_observed(counts: Counts, rng: random.Random) -> Counts:
    """
    Nonparametric bootstrap within one categorical sample.
    
    Expand into N observations, sample N with replacement, recount.
    """
    items: List[str] = []
    for k, c in counts.items():
        if c > 0:
            items.extend([k] * c)

    N = len(items)
    if N == 0:
        return {}

    sampled = [rng.choice(items) for _ in range(N)]
    out: Counts = {}
    for s in sampled:
        out[s] = out.get(s, 0) + 1
    return out


def compare_datasets(
    P: Dataset,
    Q: Dataset,
    *,
    alpha: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Non-bootstrapped metrics across shared keys (legacy, counts-based).
    
    Args:
        P: Reference dataset (counts)
        Q: Candidate dataset (counts)
        alpha: Dirichlet smoothing parameter
    
    Returns:
        Dict per metric with keys: mean, median, std
    """
    keys = list(P.keys() & Q.keys())
    if not keys:
        raise ValueError("No overlapping keys between P and Q")

    values: Dict[str, List[float]] = {
        "native_kl": [],
        "epsilon_kl": [],
        "dirichlet_kl": [],
        "dirichlet_js": [],
        "dirichlet_js_sqrt": [],
    }

    for k in keys:
        Pc = P[k]
        Qc = Q[k]

        # Native KL is diagnostic and can be inf
        try:
            nk = kl_divergence_native_from_counts(Pc, Qc)
        except Exception:
            nk = float("inf")
        ek = kl_divergence_epsilon_from_counts(Pc, Qc)
        dkl = kl_divergence_dirichlet_from_counts(Pc, Qc, alpha=alpha)
        js = js_divergence_dirichlet_from_counts(Pc, Qc, alpha=alpha, sqrt=False)

        values["native_kl"].append(nk)
        values["epsilon_kl"].append(ek)
        values["dirichlet_kl"].append(dkl)
        values["dirichlet_js"].append(js)
        values["dirichlet_js_sqrt"].append(math.sqrt(js))

    out: Dict[str, Dict[str, float]] = {}
    for m, xs in values.items():
        out[m] = {
            "mean": _mean(xs),
            "median": _percentile(xs, 0.5),
            "std": _std(xs),
        }
    return out


def compare_datasets_bootstrap(
    P: Dataset,
    Q: Dataset,
    *,
    alpha: float = 1.0,
    n_boot: int = 1000,
    ci: float = 0.95,
    mode: Mode = "both",
    seed: Optional[int] = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrapped comparison of two datasets of count distributions (legacy).

    Args:
        P: Reference dataset (counts)
        Q: Candidate dataset (counts)
        alpha: Dirichlet smoothing parameter
        n_boot: Number of bootstrap replicates
        ci: Confidence interval level
        mode: "keys", "within", or "both" (recommended)
        seed: Random seed

    Returns per metric:
        {metric: {"point": ..., "boot_mean": ..., "ci_lower": ..., "ci_upper": ...}}
    """
    rng = random.Random(seed)

    keys = list(P.keys() & Q.keys())
    if not keys:
        raise ValueError("No overlapping keys between P and Q")

    point_stats = compare_datasets(P, Q, alpha=alpha)

    metrics = list(point_stats.keys())
    boot_samples: Dict[str, List[float]] = {m: [] for m in metrics}

    pairs = [(k, P[k], Q[k]) for k in keys]
    n_keys = len(pairs)

    for _ in range(n_boot):
        if mode in ("keys", "both"):
            chosen = [pairs[rng.randrange(n_keys)] for _ in range(n_keys)]
        else:
            chosen = pairs

        totals = {m: 0.0 for m in metrics}
        count = 0

        for _, Pc, Qc in chosen:
            if mode in ("within", "both"):
                Pc_b = _resample_counts_from_observed(Pc, rng)
                Qc_b = _resample_counts_from_observed(Qc, rng)
            else:
                Pc_b, Qc_b = Pc, Qc

            try:
                nk = kl_divergence_native_from_counts(Pc_b, Qc_b)
            except Exception:
                nk = float("inf")

            ek = kl_divergence_epsilon_from_counts(Pc_b, Qc_b)
            dkl = kl_divergence_dirichlet_from_counts(Pc_b, Qc_b, alpha=alpha)
            js = js_divergence_dirichlet_from_counts(Pc_b, Qc_b, alpha=alpha, sqrt=False)
            js_sqrt = math.sqrt(js)

            totals["native_kl"] += nk
            totals["epsilon_kl"] += ek
            totals["dirichlet_kl"] += dkl
            totals["dirichlet_js"] += js
            totals["dirichlet_js_sqrt"] += js_sqrt

            count += 1

        for m in metrics:
            boot_samples[m].append(totals[m] / count)

    alpha_ci = (1.0 - ci) / 2.0
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        draws = boot_samples[m]
        out[m] = {
            "point": point_stats[m],
            "boot_mean": _mean(draws),
            "boot_median": _percentile(draws, 0.5),
            "ci_lower": _percentile(draws, alpha_ci),
            "ci_upper": _percentile(draws, 1.0 - alpha_ci),
        }

    return out
