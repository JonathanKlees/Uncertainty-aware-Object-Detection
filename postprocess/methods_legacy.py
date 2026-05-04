"""
Legacy post-processing methods returning counts.

These methods take counts and return counts (Dataset -> Dataset).
Used for backward compatibility with the legacy evaluation path.
"""

import random
from typing import Callable, Dict, List, Optional, Union

from postprocess.clever_labeling import cleverlabel
from postprocess.smoothing import Counts, Dataset, DistDataset

# Import shared configs and naming from factory (these are the same for legacy and new)
from postprocess.methods_factory import DEFAULT_METHOD_CONFIGS, params_to_name


# Type alias for legacy post-processing methods
PostProcessMethod = Callable[[Dataset], Dataset]


# =============================================================================
# Legacy counts-based functions (counts -> counts)
# =============================================================================

def resample_within_entry(
    map_id_counts: Dataset, 
    seed: int = 0
) -> Dataset:
    """
    Resample observations within each entry using bootstrap.
    
    For each object, expands counts to individual labels and resamples
    with replacement, maintaining the same total count.
    """
    rng = random.Random(seed)
    out: Dataset = {}
    
    for k, counts in map_id_counts.items():
        # Expand to labels
        labels = []
        for c, n in counts.items():
            if n > 0:
                labels.extend([c] * n)
        
        N = len(labels)
        if N == 0:
            out[k] = {}
            continue

        sampled = [rng.choice(labels) for _ in range(N)]
        new_counts: Counts = {}
        for s in sampled:
            new_counts[s] = new_counts.get(s, 0) + 1
        out[k] = new_counts
    
    return out


def downweight_proposed_class(
    map_id_counts: Dataset,
    map_id_proposed_class: Dict[str, str],
    downweight_factor: float = 0.9,
) -> Dataset:
    """
    Downweight the count of the proposed class by a factor.
    
    Args:
        map_id_counts: Object ID -> counts mapping
        map_id_proposed_class: Object ID -> proposed class mapping
        downweight_factor: Factor to multiply proposed class count by (0-1)
    """
    out: Dataset = {}
    
    for k, counts in map_id_counts.items():
        proposed_cls = map_id_proposed_class.get(k)
        if proposed_cls is None:
            out[k] = counts
            continue
        
        new_counts: Counts = {}
        for cls, n in counts.items():
            if cls == proposed_cls:
                new_n = int(n * downweight_factor)
            else:
                new_n = n
            new_counts[cls] = new_n
        out[k] = new_counts
    
    return out


def apply_cleverlabel(
    map_id_counts: Dataset,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    delta: float = 0.1,
    mu: float = 0.75,
    one_star: float = 0.99,
    avoid_overcorrection_threshold: float = 0,
    return_probs: bool = False,
) -> Union[Dataset, DistDataset]:
    """
    Apply CleverLabel bias correction to all entries (counts version).
    
    Args:
        map_id_counts: Object ID -> counts mapping
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix for class blending
        delta: Bias correction parameter (acceptance delta)
        mu: Class blending weight (1.0 = no blending, 0.0 = full blending)
        one_star: One-star acceptance probability
        avoid_overcorrection_threshold: Threshold to avoid overcorrection
        return_probs: If True, return probability distributions; if False, return counts
    
    Returns:
        Dataset (counts) if return_probs=False, DistDataset (probs) if return_probs=True
    """
    out: Union[Dataset, DistDataset] = {}
    
    for k, counts in map_id_counts.items():
        proposed_cls = map_id_proposed_class.get(k)
        if proposed_cls is None:
            if return_probs:
                # Convert counts to probs without correction
                total = sum(counts.values())
                if total > 0:
                    out[k] = {c: counts.get(c, 0) / total for c in classes}
                else:
                    out[k] = {c: 0.0 for c in classes}
            else:
                out[k] = counts
            continue

        result = cleverlabel(
            counts, 
            proposal_class=proposed_cls, 
            classes=classes, 
            delta=delta, 
            one_star=one_star, 
            mu=mu, 
            avoid_overcorrection_threshold=avoid_overcorrection_threshold,
            transition_c=transition_c,
            apply_bc=True if mu > 0 else False, 
            apply_cb=True if mu < 1 else False,
            return_counts=not return_probs,
            input_probs=False,  # Counts input
        )
        out[k] = result
    
    return out


# =============================================================================
# Legacy factory functions (counts -> counts)
# =============================================================================

def create_cleverlabel_legacy(
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    delta: float = 0.1,
    mu: float = 0.75,
    one_star: float = 0.99,
    avoid_overcorrection_threshold: float = 0,
) -> PostProcessMethod:
    """Create a CleverLabel method that returns counts."""
    def method(map_id_counts: Dataset) -> Dataset:
        return apply_cleverlabel(
            map_id_counts,
            map_id_proposed_class,
            classes,
            transition_c,
            delta=delta,
            mu=mu,
            one_star=one_star,
            avoid_overcorrection_threshold=avoid_overcorrection_threshold,
            return_probs=False,
        )
    return method


def create_downweight_legacy(
    map_id_proposed_class: Dict[str, str],
    downweight_factor: float = 0.9,
) -> PostProcessMethod:
    """Create a downweight method that returns counts."""
    def method(map_id_counts: Dataset) -> Dataset:
        return downweight_proposed_class(
            map_id_counts,
            map_id_proposed_class,
            downweight_factor=downweight_factor,
        )
    return method


def create_identity_legacy() -> PostProcessMethod:
    """Create an identity method (no transformation)."""
    def method(map_id_counts: Dataset) -> Dataset:
        return map_id_counts
    return method


def create_perfect_legacy(
    map_id_unbiased_counts: Dataset,
) -> PostProcessMethod:
    """Create a "perfect" method that returns ground truth counts."""
    def method(map_id_counts: Dataset) -> Dataset:
        return map_id_unbiased_counts
    return method


def create_resample_legacy(
    seed: int = 0,
) -> PostProcessMethod:
    """Create a resample method that resamples within each entry."""
    def method(map_id_counts: Dataset) -> Dataset:
        return resample_within_entry(map_id_counts, seed=seed)
    return method


def create_method_legacy(
    method_type: str,
    params: Dict,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_counts: Optional[Dataset] = None,
) -> PostProcessMethod:
    """
    Create a legacy method (counts -> counts) from type and parameters.
    
    Args:
        method_type: One of "identity", "perfect", "cleverlabel", "downweight", "resample"
        params: Method-specific parameters
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix
        map_id_unbiased_counts: Ground truth (for "perfect" method)
    
    Returns:
        Method function: Dataset -> Dataset
    """
    if method_type == "identity":
        return create_identity_legacy()
    
    elif method_type == "perfect":
        return create_perfect_legacy(map_id_unbiased_counts)
    
    elif method_type == "cleverlabel":
        return create_cleverlabel_legacy(
            map_id_proposed_class,
            classes,
            transition_c,
            delta=params.get("delta", 0.1),
            mu=params.get("mu", 0.75),
            one_star=params.get("one_star", 0.99),
            avoid_overcorrection_threshold=params.get("avoid_overcorrection_threshold", 0.0),
        )
    
    elif method_type == "downweight":
        return create_downweight_legacy(
            map_id_proposed_class,
            downweight_factor=params.get("downweight_factor", 0.9),
        )
    
    elif method_type == "resample":
        return create_resample_legacy(
            seed=params.get("seed", 0),
        )
    
    else:
        raise ValueError(f"Unknown method type: {method_type}")


# =============================================================================
# Default methods for legacy path
# =============================================================================


def get_default_methods_legacy(
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_counts: Optional[Dataset] = None,
) -> List[tuple]:
    """
    Get the default list of legacy post-processing methods (counts -> counts).
    
    Args:
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix for class blending
        map_id_unbiased_counts: Ground truth unbiased counts (for "perfect" baseline)
    
    Returns:
        List of (name, method) tuples where method is Dataset -> Dataset
    """
    methods = []
    for method_type, params in DEFAULT_METHOD_CONFIGS:
        name = params_to_name(method_type, params)
        method = create_method_legacy(
            method_type, params, map_id_proposed_class, classes, transition_c, map_id_unbiased_counts
        )
        methods.append((name, method))
    return methods
