"""
Factory functions, parameter grids, and tuning utilities for post-processing methods.
"""

import itertools
from typing import Callable, Dict, List, Optional, Tuple

from postprocess.smoothing import Dataset, DistDataset, Smoothing
from postprocess.methods import ProbsMethod


# =============================================================================
# Default method configurations (single source of truth)
# =============================================================================

DEFAULT_METHOD_CONFIGS: List[Tuple[str, Dict]] = [
    # identity and perfect baselines
    ("identity", {}),
    ("perfect", {}),
    ("downweight", {"downweight_factor": 0.5}),
    # Per-class damper methods (params set dynamically from training data)
    ("per_class_damper_full", {}),
    ("per_class_damper_remaining", {}),
    # sanity check no change
    ("cleverlabel", {"delta": 0.00, "mu": 1.00}),
    # only bias correction
    ("cleverlabel", {"delta": 0.05, "mu": 1.00}),
    ("cleverlabel", {"delta": 0.05, "mu": 1.00, "avoid_overcorrection_threshold": 0.5}),
    ("cleverlabel", {"delta": 0.05, "mu": 1.00, "avoid_overcorrection_threshold": 0.8}),
    ("cleverlabel", {"delta": 0.10, "mu": 1.00}),
    ("cleverlabel", {"delta": 0.20, "mu": 1.00}),
    ("cleverlabel", {"delta": 0.20, "mu": 1.00, "avoid_overcorrection_threshold": 0.5}),
    ("cleverlabel", {"delta": 0.20, "mu": 1.00, "avoid_overcorrection_threshold": 0.8}),
    ("cleverlabel", {"delta": 0.40, "mu": 1.00}),
    ("cleverlabel", {"delta": 0.60, "mu": 1.00}),
    # only class blending
    ("cleverlabel", {"delta": 0.05, "mu": 0.00}),
    # mixed
    # ("cleverlabel", {"delta": 0.05, "mu": 0.95}),
    # ("cleverlabel", {"delta": 0.05, "mu": 0.85}),
    ("cleverlabel", {"delta": 0.05, "mu": 0.75}),
]


# Smoothing variants for the new (probs-based) methods
# Each tuple: (smoothing_type, smoothing_params, name_suffix)
DEFAULT_SMOOTHING_VARIANTS = [
    # (Smoothing.DIRICHLET_POST, {"alpha": 0.1}, "dirichlet_post"),
    # (Smoothing.DIRICHLET_PRE, {"alpha": 0.1}, "dirichlet_pre"),
    (Smoothing.EPSILON, {"epsilon": 1e-10}, "epsilon_1e-10"),
    # (Smoothing.EPSILON, {"epsilon": 1e-5}, "epsilon_1e-5"),
]

# Smoothing parameter grid for tuning
SMOOTHING_PARAM_GRID = {
    "smoothing": [Smoothing.DIRICHLET_POST, Smoothing.DIRICHLET_PRE, Smoothing.EPSILON],
    "alpha": [0.1, 0.5],  # For Dirichlet smoothing
    "epsilon": [1e-10],  # For Epsilon smoothing
}

# Reduced smoothing grid for faster tuning
SMOOTHING_PARAM_GRID_FAST = {
    "smoothing": [Smoothing.EPSILON],
    "alpha": [0.5],
    "epsilon": [1e-10],
}


# =============================================================================
# Naming utilities
# =============================================================================

def params_to_name(method_type: str, params: Dict) -> str:
    """Convert method type and parameter dict to a readable name string."""
    if method_type == "identity":
        return "identity"
    
    if method_type == "perfect":
        return "perfect"
    
    if method_type == "resample":
        seed = params.get("seed", 0)
        return f"resample_s{seed}"
    
    if method_type == "downweight":
        factor = params.get("downweight_factor", 0.9)
        return f"downweight_{factor:.2f}"
    
    if method_type == "per_class_damper_full":
        return "per_class_damper_full"
    
    if method_type == "per_class_damper_remaining":
        return "per_class_damper_remaining"
    
    if method_type == "cleverlabel":
        parts = []
        if "delta" in params:
            parts.append(f"d{params['delta']:.2f}")
        if "mu" in params:
            parts.append(f"mu{params['mu']:.2f}")
        if "avoid_overcorrection_threshold" in params and params["avoid_overcorrection_threshold"] > 0:
            parts.append(f"ao{params['avoid_overcorrection_threshold']:.1f}")
        return "cleverlabel_" + "_".join(parts)
    
    return f"{method_type}_{params}"


# =============================================================================
# Method creation from config
# =============================================================================

def create_method_probs(
    method_type: str,
    params: Dict,
    smoothing: Smoothing,
    smoothing_params: Dict,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_counts: Optional[Dataset] = None,
) -> Callable[[Dataset], DistDataset]:
    """
    Create a method that returns probabilities (counts -> probs) from config.
    
    Supports all method types: identity, perfect, cleverlabel, downweight, resample.
    
    Args:
        method_type: One of "identity", "perfect", "cleverlabel", "downweight", "resample"
        params: Method-specific parameters
        smoothing: Type of smoothing to apply
        smoothing_params: Smoothing parameters (alpha or epsilon)
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix
        map_id_unbiased_counts: Ground truth (for "perfect" method)
    
    Returns:
        Method function: Dataset -> DistDataset
    """
    from postprocess.methods_wrappers import create_method_with_smoothing
    
    alpha = smoothing_params.get("alpha", 0.5)
    epsilon = smoothing_params.get("epsilon", 1e-10)
    
    return create_method_with_smoothing(
        smoothing=smoothing,
        method_type=method_type,
        params=params,
        map_id_proposed_class=map_id_proposed_class,
        classes=classes,
        transition_c=transition_c,
        map_id_unbiased_counts=map_id_unbiased_counts,
        alpha=alpha,
        epsilon=epsilon,
    )


def create_method_from_params(
    method_type: str,
    params: Dict,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_counts: Optional[Dataset] = None,
) -> Callable[[Dataset], Dataset]:
    """
    Create a legacy method (counts -> counts) from type and parameters.
    
    This is a convenience wrapper that delegates to methods_legacy.create_method_legacy.
    Use this when you need legacy behavior from the factory module.
    
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
    from postprocess.methods_legacy import create_method_legacy
    return create_method_legacy(
        method_type=method_type,
        params=params,
        map_id_proposed_class=map_id_proposed_class,
        classes=classes,
        transition_c=transition_c,
        map_id_unbiased_counts=map_id_unbiased_counts,
    )


# =============================================================================
# Get default methods
# =============================================================================

def get_default_methods(
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_counts: Optional[Dataset] = None,
    smoothing_variants: Optional[List[Tuple]] = None,
    per_class_dampers: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, Callable]]:
    """
    Get the default list of post-processing methods (counts -> probs).
    
    For legacy methods (counts -> counts), use get_default_methods_legacy from methods_legacy.
    
    Args:
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix for class blending
        map_id_unbiased_counts: Ground truth unbiased counts (for "perfect" baseline)
        smoothing_variants: Custom smoothing variants (default: DEFAULT_SMOOTHING_VARIANTS)
        per_class_dampers: Dict mapping class -> damper weight (for per_class_damper methods)
    
    Returns:
        List of (name, method) tuples where method is Dataset -> DistDataset
    """
    if smoothing_variants is None:
        smoothing_variants = DEFAULT_SMOOTHING_VARIANTS
    
    if per_class_dampers is None:
        per_class_dampers = {c: 1.0 for c in classes}  # Default to no change
    
    methods = []
    for method_type, params in DEFAULT_METHOD_CONFIGS:
        # Inject per_class_dampers into params for damper methods
        if method_type in ("per_class_damper_full", "per_class_damper_remaining"):
            params = {**params, "per_class_dampers": per_class_dampers}
        
        base_name = params_to_name(method_type, params)
        
        for smoothing, smoothing_params, suffix in smoothing_variants:
            name = f"{base_name}_{suffix}"
            method = create_method_probs(
                method_type, params, smoothing, smoothing_params,
                map_id_proposed_class, classes, transition_c, map_id_unbiased_counts
            )
            methods.append((name, method))
    
    return methods


# =============================================================================
# Parameter grids for tuning
# =============================================================================

CLEVERLABEL_PARAM_GRID = {
    "delta": [0.0, 0.05, 0.1, 0.15,  0.2, 0.3, 0.4, 0.5, 0.6],
    "mu": [0.0, 0.25, 0.5, 0.75,1.0],
    "avoid_overcorrection_threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9],
}

DOWNWEIGHT_PARAM_GRID = {
    "downweight_factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
}

RESAMPLE_PARAM_GRID = {
    "seed": [0, 1, 2, 3, 4],
}

DEFAULT_TUNING_GRID = [
    {"method": "identity"},
    # {"method": "resample", "params": RESAMPLE_PARAM_GRID},
    {"method": "downweight", "params": DOWNWEIGHT_PARAM_GRID},
    # {"method": "per_class_damper_full"},
    # {"method": "per_class_damper_remaining"},
    {"method": "cleverlabel", "params": CLEVERLABEL_PARAM_GRID},
]


# =============================================================================
# Grid utilities
# =============================================================================

def generate_param_grid(param_grid: Dict[str, List]) -> List[Dict]:
    """
    Generate all combinations of parameters from a grid.
    
    Args:
        param_grid: Dict of parameter name -> list of values
        
    Returns:
        List of parameter dicts, one for each combination
    """
    if not param_grid:
        return [{}]
    
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def get_tuning_methods(
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    tuning_grid: Optional[List[Dict]] = None,
    legacy: bool = True,
) -> List[Tuple[str, Callable, Dict]]:
    """
    Get list of methods with all parameter combinations for tuning.
    
    Args:
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix for class blending
        tuning_grid: List of method configs, each with "method" and optional "params"
                     Defaults to DEFAULT_TUNING_GRID
        legacy: If True, return legacy methods (counts -> counts)
    
    Returns:
        List of (name, method, full_params) tuples where full_params includes method type
    """
    if tuning_grid is None:
        tuning_grid = DEFAULT_TUNING_GRID
    
    methods = []
    
    for method_config in tuning_grid:
        method_type = method_config["method"]
        param_grid = method_config.get("params", {})
        
        combinations = generate_param_grid(param_grid)
        
        for params in combinations:
            name = params_to_name(method_type, params)
            method = create_method_from_params(
                method_type,
                params,
                map_id_proposed_class,
                classes,
                transition_c,
            )
            full_params = {"method": method_type, **params}
            methods.append((name, method, full_params))
    
    return methods


def _smoothing_to_name(smoothing: Smoothing, alpha: float, epsilon: float) -> str:
    """Convert smoothing parameters to a name suffix."""
    if smoothing == Smoothing.EPSILON:
        return f"eps_{epsilon:.0e}"
    else:
        return f"{smoothing.name.lower()}_a{alpha}"


def get_tuning_methods_probs(
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    map_id_unbiased_counts: Optional[Dataset] = None,
    tuning_grid: Optional[List[Dict]] = None,
    smoothing_grid: Optional[Dict] = None,
    per_class_dampers: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, Callable, Dict]]:
    """
    Get list of probs-based methods with all parameter × smoothing combinations for tuning.
    
    Args:
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix for class blending
        map_id_unbiased_counts: Ground truth unbiased counts (for "perfect" method)
        tuning_grid: List of method configs (defaults to DEFAULT_TUNING_GRID)
        smoothing_grid: Dict with smoothing params (defaults to SMOOTHING_PARAM_GRID_FAST)
        per_class_dampers: Dict mapping class -> damper weight (for per_class_damper methods)
    
    Returns:
        List of (name, method, full_params) tuples where:
        - name: Human-readable method name including smoothing suffix
        - method: Callable (Dataset -> DistDataset)
        - full_params: Dict with method, smoothing, alpha, epsilon keys
    """
    if tuning_grid is None:
        tuning_grid = DEFAULT_TUNING_GRID
    if smoothing_grid is None:
        smoothing_grid = SMOOTHING_PARAM_GRID_FAST
    if per_class_dampers is None:
        per_class_dampers = {c: 1.0 for c in classes}  # Default to no change
    
    methods = []
    
    # Generate all smoothing combinations
    smoothing_combos = []
    for smoothing in smoothing_grid.get("smoothing", [Smoothing.DIRICHLET_POST]):
        if smoothing == Smoothing.EPSILON:
            for epsilon in smoothing_grid.get("epsilon", [1e-10]):
                smoothing_combos.append((smoothing, 0.5, epsilon))
        else:
            for alpha in smoothing_grid.get("alpha", [0.5]):
                smoothing_combos.append((smoothing, alpha, 1e-10))
    
    # Generate all method × smoothing combinations
    for method_config in tuning_grid:
        method_type = method_config["method"]
        param_grid = method_config.get("params", {})
        
        param_combinations = generate_param_grid(param_grid)
        
        for params in param_combinations:
            # Inject per_class_dampers into params for damper methods
            if method_type in ("per_class_damper_full", "per_class_damper_remaining"):
                params = {**params, "per_class_dampers": per_class_dampers}
            
            base_name = params_to_name(method_type, params)
            
            for smoothing, alpha, epsilon in smoothing_combos:
                smoothing_suffix = _smoothing_to_name(smoothing, alpha, epsilon)
                name = f"{base_name}_{smoothing_suffix}"
                
                method = create_method_probs(
                    method_type=method_type,
                    params=params,
                    smoothing=smoothing,
                    smoothing_params={"alpha": alpha, "epsilon": epsilon},
                    map_id_proposed_class=map_id_proposed_class,
                    classes=classes,
                    transition_c=transition_c,
                    map_id_unbiased_counts=map_id_unbiased_counts,
                )
                
                full_params = {
                    "method": method_type,
                    "smoothing": smoothing.name,
                    "alpha": alpha,
                    "epsilon": epsilon,
                    **params,
                }
                methods.append((name, method, full_params))
    
    return methods
