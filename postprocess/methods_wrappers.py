"""
Smoothing-aware wrappers for post-processing methods.

Architecture:
- DIRICHLET_POST: counts → normalize → probs → PP(probs→probs) → to pseudo-counts (original total) → Dirichlet → probs
- DIRICHLET_PRE: counts → Dirichlet → probs → PP(probs→probs) → probs
- EPSILON: counts → normalize → probs → PP(probs→probs) → epsilon smooth → probs
- Legacy: counts → legacy PP(counts→counts) → counts (smoothing in legacy metrics)

All smoothed methods take counts as input and return probability distributions.
"""

from typing import Callable, Dict, List, Optional

from postprocess.smoothing import (
    Smoothing,
    Dataset,
    DistDataset,
    Distribution,
    counts_to_dirichlet_probs,
    epsilon_smooth_probs,
)


def normalize_counts_to_probs(counts: Dict[str, int], classes: List[str]) -> Distribution:
    """Convert counts to normalized probabilities."""
    total = sum(counts.values())
    if total > 0:
        return {c: counts.get(c, 0) / total for c in classes}
    else:
        return {c: 0.0 for c in classes}


def probs_to_pseudo_counts(probs: Distribution, total: int, classes: List[str]) -> Dict[str, int]:
    """
    Convert probabilities to pseudo-counts preserving total exactly.
    
    Uses largest-remainder rounding from smoothing.probs_to_counts.
    """
    from postprocess.smoothing import probs_to_counts
    return probs_to_counts(probs, total, classes)


def create_probs_method(
    method_type: str,
    params: Dict,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_probs: Optional[DistDataset] = None,
) -> Callable[[DistDataset], DistDataset]:
    """
    Create a probs-based post-processing method (probs → probs).
    
    Args:
        method_type: Method type (identity, perfect, cleverlabel, downweight, resample)
        params: Method parameters
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix
        map_id_unbiased_probs: Ground truth probs (for "perfect" method)
    
    Returns:
        Method function: DistDataset → DistDataset
    """
    from postprocess.methods import (
        identity,
        perfect,
        downweight_proposed_class,
        apply_cleverlabel,
        resample,
    )
    
    if method_type == "identity":
        return identity
    
    elif method_type == "perfect":
        def perfect_method(map_id_probs: DistDataset) -> DistDataset:
            return perfect(map_id_probs, map_id_unbiased_probs)
        return perfect_method
    
    elif method_type == "downweight":
        def downweight_method(map_id_probs: DistDataset) -> DistDataset:
            return downweight_proposed_class(
                map_id_probs,
                map_id_proposed_class,
                classes,
                downweight_factor=params.get("downweight_factor", 0.9),
            )
        return downweight_method
    
    elif method_type == "cleverlabel":
        def cleverlabel_method(map_id_probs: DistDataset) -> DistDataset:
            return apply_cleverlabel(
                map_id_probs,
                map_id_proposed_class,
                classes,
                transition_c,
                delta=params.get("delta", 0.1),
                mu=params.get("mu", 0.75),
                one_star=params.get("one_star", 0.99),
                avoid_overcorrection_threshold=params.get("avoid_overcorrection_threshold", 0.0),
            )
        return cleverlabel_method
    
    elif method_type == "resample":
        def resample_method(map_id_probs: DistDataset) -> DistDataset:
            return resample(
                map_id_probs,
                classes,
                n_samples=params.get("n_samples", 100),
                seed=params.get("seed", 0),
            )
        return resample_method
    
    elif method_type == "per_class_damper_full":
        from postprocess.methods import apply_per_class_damper
        per_class_dampers = params.get("per_class_dampers", {})
        def per_class_damper_full_method(map_id_probs: DistDataset) -> DistDataset:
            return apply_per_class_damper(
                map_id_probs,
                map_id_proposed_class,
                classes,
                per_class_dampers,
                renorm_mode="full",
            )
        return per_class_damper_full_method
    
    elif method_type == "per_class_damper_remaining":
        from postprocess.methods import apply_per_class_damper
        per_class_dampers = params.get("per_class_dampers", {})
        def per_class_damper_remaining_method(map_id_probs: DistDataset) -> DistDataset:
            return apply_per_class_damper(
                map_id_probs,
                map_id_proposed_class,
                classes,
                per_class_dampers,
                renorm_mode="remaining",
            )
        return per_class_damper_remaining_method
    
    else:
        raise ValueError(f"Unknown method type: {method_type}")


def create_method_with_smoothing(
    smoothing: Smoothing,
    method_type: str,
    params: Dict,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_counts: Optional[Dataset] = None,
    alpha: float = 0.5,
    epsilon: float = 1e-10,
) -> Callable[[Dataset], DistDataset]:
    """
    Create a post-processing method with smoothing applied.
    
    Architecture:
    - DIRICHLET_POST: counts → normalize → probs → PP → to pseudo-counts (original total) → Dirichlet → probs
    - DIRICHLET_PRE: counts → Dirichlet → probs → PP → probs
    - EPSILON: counts → normalize → probs → PP → epsilon smooth → probs
    
    Args:
        smoothing: Type of smoothing pipeline
        method_type: Method type (identity, perfect, cleverlabel, downweight, resample)
        params: Method parameters
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix
        map_id_unbiased_counts: Ground truth unbiased counts (for "perfect" method)
        alpha: Dirichlet smoothing parameter
        epsilon: Epsilon smoothing parameter
    
    Returns:
        Method function that takes counts and returns probability distributions
    """
    # Convert unbiased counts to probs for perfect method
    map_id_unbiased_probs = None
    if map_id_unbiased_counts is not None:
        map_id_unbiased_probs = {
            k: normalize_counts_to_probs(c, classes)
            for k, c in map_id_unbiased_counts.items()
        }
    
    # Create probs-based method (probs -> probs)
    probs_method = create_probs_method(
        method_type, params, map_id_proposed_class, classes, transition_c, map_id_unbiased_probs
    )
    
    if smoothing == Smoothing.DIRICHLET_POST:
        # counts → normalize → probs → PP → to pseudo-counts (original total) → Dirichlet → probs
        def dirichlet_post_method(map_id_counts: Dataset) -> DistDataset:
            # Step 1: normalize counts to probs
            map_id_probs = {
                k: normalize_counts_to_probs(c, classes)
                for k, c in map_id_counts.items()
            }
            # Step 2: apply probs-based PP
            pp_probs = probs_method(map_id_probs)
            # Step 3: convert to pseudo-counts (using original total) and apply Dirichlet
            out = {}
            for k, probs in pp_probs.items():
                original_total = sum(map_id_counts[k].values()) if k in map_id_counts else 0
                if original_total > 0:
                    pseudo_counts = probs_to_pseudo_counts(probs, original_total, classes)
                    out[k] = counts_to_dirichlet_probs(pseudo_counts, alpha=alpha, support=classes)
                else:
                    out[k] = counts_to_dirichlet_probs({}, alpha=alpha, support=classes)
            return out
        return dirichlet_post_method
    
    elif smoothing == Smoothing.DIRICHLET_PRE:
        # counts → Dirichlet → probs → PP → probs
        def dirichlet_pre_method(map_id_counts: Dataset) -> DistDataset:
            # Step 1: apply Dirichlet smoothing to get probs
            map_id_probs = {
                k: counts_to_dirichlet_probs(c, alpha=alpha, support=classes)
                for k, c in map_id_counts.items()
            }
            # Step 2: apply probs-based PP
            return probs_method(map_id_probs)
        return dirichlet_pre_method
    
    elif smoothing == Smoothing.EPSILON:
        # counts → normalize → probs → PP → epsilon smooth → probs
        def epsilon_method(map_id_counts: Dataset) -> DistDataset:
            # Step 1: normalize counts to probs
            map_id_probs = {
                k: normalize_counts_to_probs(c, classes)
                for k, c in map_id_counts.items()
            }
            # Step 2: apply probs-based PP
            pp_probs = probs_method(map_id_probs)
            # Step 3: apply epsilon smoothing
            return {
                k: epsilon_smooth_probs(p, epsilon=epsilon)
                for k, p in pp_probs.items()
            }
        return epsilon_method
    
    else:
        raise ValueError(f"Unknown smoothing: {smoothing}")
