#!/usr/bin/env python3
"""
Automatic parameter tuning for post-processing methods.

Searches over method parameters AND smoothing parameters to find optimal configuration.

Workflow:
1. Load dataset and split into train/validation sets
2. Search over parameter grid (method params × smoothing params) on train set
3. Select best parameters based on chosen metric
4. Evaluate on validation set
5. Optionally save config for later application

Usage:
    # Use shorthand dataset names:
    python -m postprocess.run_tuning --dataset cityscapes
    python -m postprocess.run_tuning --dataset coco_kitchen --save_config configs/coco_kitchen.json
    
    # Use full smoothing grid (slower but more thorough):
    python -m postprocess.run_tuning --dataset kitti --full_smoothing_grid
"""

import argparse
import json
import os
from os.path import join
from typing import Dict, List, Optional, Tuple

from postprocess.clever_labeling import get_tranistion_matrix
from postprocess.data_loader import (
    apply_cantsolve_multiplier,
    create_distributions,
    load_dataset_from_json,
    filter_to_common_keys,
    get_available_datasets,
    find_dataset_path,
    split_dataset,
    resolve_dataset_name,
    DATASET_SHORTCUTS,
)
from postprocess.evaluation import (
    compute_per_class_dampers,
    evaluate_method,
    print_evaluation_results,
    print_baseline_comparison,
)
from postprocess.methods_factory import (
    get_tuning_methods_probs,
    create_method_probs,
    params_to_name,
    DEFAULT_TUNING_GRID,
    SMOOTHING_PARAM_GRID,
    SMOOTHING_PARAM_GRID_FAST,
)
from postprocess.smoothing import Dataset, DistDataset, Smoothing, smooth_counts
from postprocess.metrics import Metric
from postprocess.mlp_tuning import get_mlp_method_for_tuning


def extract_metric(results: Dict, metric_path: str) -> float:
    """
    Extract a metric value from evaluation results using a dot-separated path.
    
    Examples:
        - "delta_summary.mean.point" -> results["delta_summary"]["mean"]["point"]
        - "distribution_comparison.dirichlet_js_sqrt.mean"
    """
    parts = metric_path.split(".")
    value = results
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return float("inf")
        if value is None:
            return float("inf")
    return float(value)


def tune_on_train(
    train_biased: Dataset,
    train_unbiased: Dataset,
    train_proposed: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    per_class_dampers: Optional[Dict[str, float]] = None,
    tuning_grid: Optional[List[Dict]] = None,
    smoothing_grid: Optional[Dict] = None,
    metric: str = "all_metrics.js_sqrt.mean",
    minimize: bool = True,
    verbose: bool = True,
) -> Tuple[Dict, float, str]:
    """
    Search for best parameters on training set.
    
    Searches over all combinations of method parameters and smoothing parameters.
    
    Args:
        train_biased: Training biased counts
        train_unbiased: Training unbiased counts (ground truth)
        train_proposed: Training proposed classes
        classes: List of class names
        transition_c: Transition matrix
        per_class_dampers: Per-class damper weights (computed from training data)
        tuning_grid: List of method configs to search (defaults to DEFAULT_TUNING_GRID)
        smoothing_grid: Smoothing parameter grid (defaults to SMOOTHING_PARAM_GRID_FAST)
        metric: Dot-separated path to metric for optimization (e.g., "all_metrics.js_sqrt.mean")
        minimize: If True, lower is better; if False, higher is better
        verbose: Print progress
    
    Returns:
        (best_params, best_score, best_name)
    """
    methods = get_tuning_methods_probs(
        train_proposed, classes, transition_c, 
        map_id_unbiased_counts=None,  # Only needed for "perfect" method
        tuning_grid=tuning_grid,
        smoothing_grid=smoothing_grid,
        per_class_dampers=per_class_dampers,
    )
    
    best_params = None
    best_score = float("inf") if minimize else float("-inf")
    best_name = ""
    
    if verbose:
        print(f"\nSearching {len(methods)} parameter combinations (method \u00d7 smoothing)...")
        print(f"Optimizing: {metric} ({'minimize' if minimize else 'maximize'})")
        print("-" * 60)
    
    for i, (name, method, params) in enumerate(methods):
        # Extract smoothing params from full_params
        smoothing_type = Smoothing[params["smoothing"]]
        alpha = params["alpha"]
        epsilon = params["epsilon"]
        
        # Smooth ground truth and biased with the same smoothing
        smoothing_type = Smoothing. EPSILON  # Use consistent smoothing for baseline and tuning
        epsilon = 1e-10
        train_unbiased_probs = {
            k: smooth_counts(c, smoothing_type, classes, alpha, epsilon)
            for k, c in train_unbiased.items()
        }
        train_biased_probs = {
            k: smooth_counts(c, smoothing_type, classes, alpha, epsilon)
            for k, c in train_biased.items()
        }
        
        # Apply method (returns probs)
        post_processed_probs = method(train_biased)
        
        # Evaluate using evaluate_method
        results = evaluate_method(
            train_unbiased_probs,
            train_biased_probs,
            post_processed_probs,
            train_proposed,
            metric=Metric.JS_SQRT,  # Default for delta computation
        )
        
        # Extract score using metric path
        score = extract_metric(results, metric)
        
        # Check if better
        is_better = (score < best_score) if minimize else (score > best_score)
        
        if is_better:
            best_params = params
            best_score = score
            best_name = name
            if verbose:
                print(f"  [{i+1:3d}/{len(methods)}] {name}: {score:.6f} (NEW BEST)")
        elif verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1:3d}/{len(methods)}] ... (best so far: {best_score:.6f})")
    
    if verbose:
        print("-" * 60)
        print(f"Best: {best_name} with {metric} = {best_score:.6f}")
        print(f"Parameters: {best_params}")
    
    return best_params, best_score, best_name


def save_config(
    output_path: str,
    dataset: str,
    parameters: Dict,
    transition_matrix: Dict[str, List[float]],
    classes: List[str],
    tuning_metric: str,
    tuning_score: float,
    train_ratio: float,
    seed: int,
) -> None:
    """
    Save tuning configuration to a JSON file for later use with run_apply.py.
    
    Args:
        output_path: Path to save the config JSON
        dataset: Name of the dataset used for tuning
        parameters: Best parameters found during tuning (includes smoothing params)
        transition_matrix: Class transition matrix computed from training data
        classes: List of class names
        tuning_metric: Metric used for optimization
        tuning_score: Best score achieved on train set
        train_ratio: Train/validation split ratio
        seed: Random seed used for splitting
    """
    config = {
        "dataset": dataset,
        "parameters": parameters,
        "transition_matrix": transition_matrix,
        "classes": classes,
        "tuning_metadata": {
            "metric": tuning_metric,
            "score": tuning_score,
            "train_ratio": train_ratio,
            "seed": seed,
        }
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to: {output_path}")


def evaluate_on_validation(
    best_params: Dict,
    val_biased: Dataset,
    val_unbiased: Dataset,
    val_proposed: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
) -> Dict:
    """
    Evaluate best parameters on validation set.
    
    Args:
        best_params: Best parameters dict (includes method, smoothing, alpha, epsilon)
        val_biased: Validation biased counts
        val_unbiased: Validation unbiased counts
        val_proposed: Validation proposed classes
        classes: List of class names
        transition_c: Transition matrix
    
    Returns:
        Evaluation results dict from evaluate_method
    """
    # Extract params
    method_type = best_params.get("method", "cleverlabel")
    smoothing_type = Smoothing[best_params.get("smoothing", "DIRICHLET_POST")]
    alpha = best_params.get("alpha", 0.5)
    epsilon = best_params.get("epsilon", 1e-10)
    
    # Create probs-based method
    method_params = {k: v for k, v in best_params.items() 
                     if k not in ["method", "smoothing", "alpha", "epsilon"]}
    
    method = create_method_probs(
        method_type=method_type,
        params=method_params,
        smoothing=smoothing_type,
        smoothing_params={"alpha": alpha, "epsilon": epsilon},
        map_id_proposed_class=val_proposed,
        classes=classes,
        transition_c=transition_c,
        map_id_unbiased_counts=None,
    )
    
    # Smooth ground truth and biased
    smoothing_type = Smoothing. EPSILON  # Use consistent smoothing for baseline and tuning
    epsilon = 1e-10
    val_unbiased_probs = {
        k: smooth_counts(c, smoothing_type, classes, alpha, epsilon)
        for k, c in val_unbiased.items()
    }
    val_biased_probs = {
        k: smooth_counts(c, smoothing_type, classes, alpha, epsilon)
        for k, c in val_biased.items()
    }
    
    # Apply method
    post_processed_probs = method(val_biased)
    
    # Evaluate
    results = evaluate_method(
        val_unbiased_probs,
        val_biased_probs,
        post_processed_probs,
        val_proposed,
        metric=Metric.JS_SQRT,  # Default for delta computation
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tune post-processing parameters on train set, evaluate on validation"
    )
    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "our_datasets")
    
    shortcuts_help = ", ".join(sorted(DATASET_SHORTCUTS.keys()))
    parser.add_argument(
        "--dataset", 
        required=True, 
        help=f"Dataset name or shortcut. Shortcuts: {shortcuts_help}"
    )
    parser.add_argument(
        "--data_dir", 
        default=default_data_dir,
        help="Directory containing dataset JSON files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Fraction of data for training/tuning (default: 0.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="delta_summary.mean.point",
        help="Metric path to optimize. Examples: "\
             "all_metrics.js_sqrt.mean (JS sqrt divergence), "\
             "all_metrics.kl.mean (KL divergence), "\
             "all_metrics.js.mean (JS divergence), "\
             "avg_prob_proposed.reduction (proposed class probability change), "\
             "delta_summary.mean.point (mean delta divergence)"
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize metric instead of minimizing"
    )
    parser.add_argument(
        "--fast_smoothing_grid",
        action="store_true",
        help="Use fast smoothing parameter grid (faster but less thorough)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--save_config",
        type=str,
        default=None,
        help="Path to save tuning config JSON (e.g., configs/cityscapes.json)"
    )
    parser.add_argument(
        "--use_mlp",
        action="store_true",
        help="Use MLP for post-processing (requires PyTorch)"
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=64,
        help="MLP hidden layer dimension (default: 64)"
    )
    parser.add_argument(
        "--mlp_epochs",
        type=int,
        default=100,
        help="MLP training epochs (default: 100)"
    )
    
    args = parser.parse_args()

    # Resolve shorthand dataset name and find path
    dataset_name = resolve_dataset_name(args.dataset)
    try:
        dataset_path = find_dataset_path(args.data_dir, dataset_name)
    except FileNotFoundError:
        available = get_available_datasets(args.data_dir)
        raise SystemExit(
            f"Dataset not found: {args.dataset}\n"
            f"Available datasets: {', '.join(sorted(available))}"
        )

    # Load dataset
    map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class, classes = (
        load_dataset_from_json(dataset_path)
    )

    print(f"Loaded dataset '{dataset_name}':")
    print(f"  Total objects: {len(map_id_biased_counts)}")
    print(f"  Classes: {classes}")

    # Filter to common keys
    map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class = filter_to_common_keys(
        map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class
    )
    print(f"  Objects with both annotations: {len(map_id_biased_counts)}")

    map_id_unbiased_counts = apply_cantsolve_multiplier(map_id_unbiased_counts)

    # Split into train/validation
    (train_biased, train_unbiased, train_proposed), \
    (val_biased, val_unbiased, val_proposed) = split_dataset(
        map_id_biased_counts,
        map_id_unbiased_counts,
        map_id_proposed_class,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    
    print(f"\nData split (seed={args.seed}):")
    print(f"  Train: {len(train_biased)} objects ({args.train_ratio:.0%})")
    print(f"  Validation: {len(val_biased)} objects ({1-args.train_ratio:.0%})")

    # Calculate transition matrix on TRAIN data only
    transition_c = get_tranistion_matrix(classes, train_unbiased)
    
    # Compute per-class dampers on TRAIN data (for per_class_damper methods)
    train_biased_probs = create_distributions(train_biased)
    train_unbiased_probs = create_distributions(train_unbiased)
    per_class_dampers = compute_per_class_dampers(
        train_biased_probs, train_unbiased_probs, train_proposed, classes
    )
    if not args.quiet:
        print(f"\nPer-class dampers computed from training data:")
        for cls, weight in sorted(per_class_dampers.items()):
            print(f"  {cls}: {weight:.4f}")
    
    # Train MLP before grid search (if enabled)
    mlp_result = None
    if args.use_mlp:
        print(f"\n{'='*60}")
        print("TRAINING MLP")
        print(f"{'='*60}")
        mlp_result = get_mlp_method_for_tuning(
            train_biased,
            train_unbiased,
            train_proposed,
            classes,
            transition_c,
            smoothing=Smoothing.EPSILON,
            alpha=0.5,
            epsilon=1e-10,
            hidden_dim=args.mlp_hidden_dim,
            epochs=args.mlp_epochs,
            verbose=not args.quiet,
        )
    
    # Select smoothing grid
    smoothing_grid = SMOOTHING_PARAM_GRID if not args.fast_smoothing_grid else SMOOTHING_PARAM_GRID_FAST

    # Tune on train set
    best_params, best_score, best_name = tune_on_train(
        train_biased,
        train_unbiased,
        train_proposed,
        classes,
        transition_c,
        per_class_dampers=per_class_dampers,
        tuning_grid=DEFAULT_TUNING_GRID,
        smoothing_grid=smoothing_grid,
        metric=args.metric,
        minimize=not args.maximize,
        verbose=not args.quiet,
    )

    # Evaluate on validation set
    val_results = evaluate_on_validation(
        best_params,
        val_biased,
        val_unbiased,
        val_proposed,
        classes,
        transition_c,
    )

    # Also evaluate best params on train set for comparison
    train_results = evaluate_on_validation(
        best_params,
        train_biased,
        train_unbiased,
        train_proposed,
        classes,
        transition_c,
    )

    # Get baseline JS_SQRT for percentage calculations
    # Use the smoothing from best_params for consistency
    smoothing_type = Smoothing[best_params.get("smoothing", "DIRICHLET_POST")]
    alpha = best_params.get("alpha", 0.5)
    epsilon = best_params.get("epsilon", 1e-10)

    # Print baseline comparisons
    print(f"\n{'='*60}")
    print("BASELINE: Biased vs Unbiased (Train)")
    print(f"{'='*60}")
    baseline_summary_train = print_baseline_comparison(
        train_unbiased, train_biased, classes, 
        alpha=alpha, smoothing=smoothing_type, epsilon=epsilon
    )

    print(f"\n{'='*60}")
    print("BASELINE: Biased vs Unbiased (Validation)")
    print(f"{'='*60}")
    baseline_summary_val = print_baseline_comparison(
        val_unbiased, val_biased, classes,
        alpha=alpha, smoothing=smoothing_type, epsilon=epsilon
    )

    # Print evaluation results
    baseline_js_sqrt_train = baseline_summary_train.get("js_sqrt", {}).get("mean", 0.0)
    baseline_js_sqrt_val = baseline_summary_val.get("js_sqrt", {}).get("mean", 0.0)

    print(f"\n{'='*60}")
    print(f"TRAIN RESULTS: {best_name}")
    print(f"{'='*60}")
    print_evaluation_results(best_name, train_results, baseline_js_sqrt_train)

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS: {best_name}")
    print(f"{'='*60}")
    print_evaluation_results(best_name, val_results, baseline_js_sqrt_val)

    # Evaluate MLP if trained
    mlp_val_score = None
    if mlp_result is not None:
        mlp_name, mlp_method, mlp_params = mlp_result
        
        # Evaluate MLP on train
        mlp_post_processed_train = mlp_method(train_biased)
        smoothing_type_eval = Smoothing.EPSILON
        epsilon_eval = 1e-10
        train_unbiased_probs = {
            k: smooth_counts(c, smoothing_type_eval, classes, 0.5, epsilon_eval)
            for k, c in train_unbiased.items()
        }
        train_biased_probs = {
            k: smooth_counts(c, smoothing_type_eval, classes, 0.5, epsilon_eval)
            for k, c in train_biased.items()
        }
        mlp_train_results = evaluate_method(
            train_unbiased_probs,
            train_biased_probs,
            mlp_post_processed_train,
            train_proposed,
            metric=Metric.JS_SQRT,
        )
        
        # Evaluate MLP on validation
        mlp_post_processed_val = mlp_method(val_biased)
        val_unbiased_probs = {
            k: smooth_counts(c, smoothing_type_eval, classes, 0.5, epsilon_eval)
            for k, c in val_unbiased.items()
        }
        val_biased_probs = {
            k: smooth_counts(c, smoothing_type_eval, classes, 0.5, epsilon_eval)
            for k, c in val_biased.items()
        }
        mlp_val_results = evaluate_method(
            val_unbiased_probs,
            val_biased_probs,
            mlp_post_processed_val,
            val_proposed,
            metric=Metric.JS_SQRT,
        )
        
        print(f"\n{'='*60}")
        print(f"MLP TRAIN RESULTS: {mlp_name}")
        print(f"{'='*60}")
        print_evaluation_results(mlp_name, mlp_train_results, baseline_js_sqrt_train)
        
        print(f"\n{'='*60}")
        print(f"MLP VALIDATION RESULTS: {mlp_name}")
        print(f"{'='*60}")
        print_evaluation_results(mlp_name, mlp_val_results, baseline_js_sqrt_val)
        
        mlp_val_score = extract_metric(mlp_val_results, args.metric)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best grid search parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    val_score = extract_metric(val_results, args.metric)
    print(f"\nGrid Search - Train {args.metric}: {best_score:.6f}")
    print(f"Grid Search - Validation {args.metric}: {val_score:.6f}")
    
    if mlp_val_score is not None:
        mlp_train_score = extract_metric(mlp_train_results, args.metric)
        print(f"\nMLP - Train {args.metric}: {mlp_train_score:.6f}")
        print(f"MLP - Validation {args.metric}: {mlp_val_score:.6f}")
        
        # Compare
        minimize = not args.maximize
        if minimize:
            winner = "MLP" if mlp_val_score < val_score else "Grid Search"
        else:
            winner = "MLP" if mlp_val_score > val_score else "Grid Search"
        print(f"\nBetter validation {args.metric}: {winner}")

    # Save config 
    save_config(
        output_path=join(".","configs",args.dataset + ".json"),
        dataset=dataset_name,
        parameters=best_params,
        transition_matrix=transition_c,
        classes=classes,
        tuning_metric=args.metric,
        tuning_score=best_score,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
