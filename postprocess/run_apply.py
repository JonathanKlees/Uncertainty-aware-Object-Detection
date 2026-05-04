#!/usr/bin/env python3
"""
Apply tuned post-processing parameters to soft label files.

Workflow:
1. Load config (single or multi-config for COCO)
2. Load input soft label file
3. For each annotation object:
   - Single proposal: Apply post-processing directly
   - Merged proposals: Apply separately per proposal, then average
4. Write 'postprocessed_soft_label' field to each object
5. Save updated JSON file
6. Generate statistics and visualization

Output locations:
- Main output: outputs/application/<input_name>_postprocessed.json (unless --inplace or --output)
- Statistics: outputs/application/<input_name>_stats.json (always)
- Histogram: outputs/application/<input_name>_histogram.png (always)

Usage:
    # Using shortcuts
    python -m postprocess.run_apply --config cityscapes --input soft_Cityscapes_train
    python -m postprocess.run_apply --config coco --input soft_COCO_2017_val

    # Overwrite input file in place
    python -m postprocess.run_apply --config cityscapes --input soft_Cityscapes_train --inplace

    # Using full paths
    python -m postprocess.run_apply --config configs/cityscapes.json --input our_datasets/soft_Cityscapes_train.json
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from postprocess.config_loader import (
    load_config,
    is_multi_config,
    get_config_for_class,
    validate_config,
)
from postprocess.data_loader import find_dataset_path, get_available_datasets, DATASET_SHORTCUTS
from postprocess.methods_factory import create_method_probs, params_to_name
from postprocess.smoothing import Counts, Dataset, Smoothing


def resolve_config_path(config_arg: str) -> str:
    """
    Resolve config argument to full path.
    
    If config_arg is an existing file, use it directly.
    Otherwise, construct path as configs/<config_arg>.json
    
    Args:
        config_arg: Either a shortcut (e.g., 'cityscapes') or a path
        
    Returns:
        Full path to config file
    """
    # If it's already an existing file, use it
    if os.path.isfile(config_arg):
        return config_arg
    
    # Construct path from shortcut: cityscapes -> configs/cityscapes.json
    return f"configs/{config_arg}.json"


def find_proposal_indices(obj: Dict[str, Any]) -> List[int]:
    """
    Find all proposal indices in an annotation object.
    
    Returns:
        List of indices. Empty list if single proposed_class (no suffix).
        [1, 2] if proposed_class_1 and proposed_class_2 exist.
    """
    indices = []
    for key in obj.keys():
        match = re.match(r'^proposed_class_(\d+)$', key)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def apply_postprocess_single(
    frequencies: Dict[str, int],
    proposed_class: str,
    config: Dict[str, Any],
) -> List[float]:
    """
    Apply post-processing to a single frequency distribution.
    
    Uses probs-based method that returns probabilities directly.
    
    Args:
        frequencies: Vote counts per class
        proposed_class: The proposed class for this annotation
        config: Config with parameters, transition_matrix, classes
        
    Returns:
        Post-processed probabilities as list (in class order)
    """
    classes = config["classes"]
    transition_matrix = config["transition_matrix"]
    params = config["parameters"]
    
    # Extract smoothing params from config
    method_type = params.get("method", "cleverlabel")
    smoothing_type = Smoothing[params.get("smoothing", "DIRICHLET_POST")]
    alpha = params.get("alpha", 0.5)
    epsilon = params.get("epsilon", 1e-10)
    
    # Method-specific params (exclude smoothing-related)
    method_params = {k: v for k, v in params.items()
                     if k not in ["method", "smoothing", "alpha", "epsilon"]}
    
    # Create single-object dataset for processing
    obj_id = "temp"
    map_id_counts: Dataset = {obj_id: frequencies}
    map_id_proposed: Dict[str, str] = {obj_id: proposed_class}
    
    # Create probs-based method
    method = create_method_probs(
        method_type=method_type,
        params=method_params,
        smoothing=smoothing_type,
        smoothing_params={"alpha": alpha, "epsilon": epsilon},
        map_id_proposed_class=map_id_proposed,
        classes=classes,
        transition_c=transition_matrix,
    )
    
    # Apply method - returns Dict[str, Distribution]
    result = method(map_id_counts)
    
    # Convert distribution dict to list in class order
    dist = result.get(obj_id, {})
    return [dist.get(c, 0.0) for c in classes]


def average_probs(probs_list: List[List[float]]) -> List[float]:
    """
    Average multiple probability vectors element-wise.
    
    For merged annotations, we average the post-processed probabilities.
    
    Args:
        probs_list: List of probability vectors (same length)
        
    Returns:
        Averaged probability vector
    """
    if not probs_list:
        return []
    if len(probs_list) == 1:
        return probs_list[0]
    
    n = len(probs_list)
    k = len(probs_list[0])
    return [sum(p[i] for p in probs_list) / n for i in range(k)]


def select_best_source(processed_probs_list: List[List[float]]) -> List[float]:
    """
    Select the source whose postprocessed distribution best supports the majority class.

    For merged objects where sources propose different classes, averaging can dilute
    the correct answer. Instead:
    1. Compute averaged distribution to find majority class (argmax)
    2. Select the source with highest P(majority_class)

    Args:
        processed_probs_list: Per-source postprocessed probability vectors

    Returns:
        The selected source's probability vector
    """
    if len(processed_probs_list) == 1:
        return processed_probs_list[0]

    averaged = average_probs(processed_probs_list)
    majority_idx = max(range(len(averaged)), key=lambda i: averaged[i])

    best_idx = max(range(len(processed_probs_list)),
                   key=lambda i: processed_probs_list[i][majority_idx])
    return processed_probs_list[best_idx]


def process_annotation(
    obj: Dict[str, Any],
    config: Dict[str, Any],
    multi_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], Optional[str], Optional[list[str]]]:
    """
    Process a single annotation object and return the postprocessed soft label.
    
    Args:
        obj: Annotation object with frequencies, proposed_class, etc.
        config: Default config to use
        multi_config: Optional multi-config for per-class configs
        
    Returns:
        (postprocessed_soft_label, postprocessed_selected_soft_label,
         proposed_class_used, method_name_used)
    """
    classes = config["classes"]
    
    # Check for merged annotations (proposed_class_1, proposed_class_2, ...)
    proposal_indices = find_proposal_indices(obj)
    
    if proposal_indices:
        # Merged annotations: process each proposal separately, then average
        processed_probs_list = []
        method_names_used = []
        
        for idx in proposal_indices:
            freq_key = f"frequencies_{idx}"
            prop_key = f"proposed_class_{idx}"
            
            frequencies = obj.get(freq_key, {})
            proposed_class = obj.get(prop_key)
            
            if not frequencies or not proposed_class:
                continue
            
            # Get config for this class (may differ per super-category)
            if multi_config:
                class_config = get_config_for_class(proposed_class, multi_config) or config
            else:
                class_config = config
            
            # Get method name
            params = class_config.get("parameters", {})
            method_type = params.get("method", "cleverlabel")
            method_params = {k: v for k, v in params.items() if k not in ["method", "smoothing", "alpha", "epsilon"]}
            method_name = params_to_name(method_type, method_params)
            method_names_used.append(method_name)
            
            # Apply post-processing (returns probabilities directly)
            processed_probs = apply_postprocess_single(frequencies, proposed_class, class_config)
            processed_probs_list.append(processed_probs)
        
        if not processed_probs_list:
            # Fallback to original soft_label
            fallback = obj.get("soft_label", [0.0] * len(classes))
            return fallback, fallback, None, None
        
        # Average all processed probabilities
        postprocessed_soft_label = average_probs(processed_probs_list)
        selected_soft_label = select_best_source(processed_probs_list)
        return postprocessed_soft_label, selected_soft_label, f"merged_{len(proposal_indices)}", method_names_used
    
    else:
        # Single annotation
        frequencies = obj.get("frequencies", {})
        proposed_class = obj.get("proposed_class")
        
        if not frequencies or not proposed_class:
            # Return original soft_label
            fallback = obj.get("soft_label", [0.0] * len(classes))
            return fallback, fallback, None, None
        
        # Get config for this class
        if multi_config:
            class_config = get_config_for_class(proposed_class, multi_config) or config
        else:
            class_config = config
        
        # Get method name
        params = class_config.get("parameters", {})
        method_type = params.get("method", "cleverlabel")
        method_params = {k: v for k, v in params.items() if k not in ["method", "smoothing", "alpha", "epsilon"]}
        method_name = params_to_name(method_type, method_params)
        
        # Apply post-processing (returns probabilities directly)
        postprocessed_soft_label = apply_postprocess_single(frequencies, proposed_class, class_config)
        return postprocessed_soft_label, postprocessed_soft_label, proposed_class, [method_name]


def compute_statistics(
    data: Dict[str, Any],
    classes: List[str],
) -> Dict[str, Any]:
    """
    Compute statistics comparing original and postprocessed soft labels.
    
    Returns dict with:
    - prob_proposed_before: mean/median/std of P(proposed_class) before
    - prob_proposed_after: mean/median/std of P(proposed_class) after
    - num_objects: total number of objects processed
    - per_class_stats: statistics broken down by proposed class
    """
    import statistics
    
    prob_before = []
    prob_after = []
    per_class_before: Dict[str, List[float]] = {}
    per_class_after: Dict[str, List[float]] = {}
    
    objects = data.get("objects", {})
    
    for img_name, obj_list in objects.items():
        for obj in obj_list:
            soft_label = obj.get("soft_label", [])
            postprocessed = obj.get("postprocessed_soft_label", [])
            proposed = obj.get("proposed_class")
            
            if proposed is None or proposed not in classes:
                continue
            
            idx = classes.index(proposed)
            if idx < len(soft_label):
                p_before = soft_label[idx]
                prob_before.append(p_before)
                per_class_before.setdefault(proposed, []).append(p_before)
            
            if idx < len(postprocessed):
                p_after = postprocessed[idx]
                prob_after.append(p_after)
                per_class_after.setdefault(proposed, []).append(p_after)
    
    def safe_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }
    
    stats = {
        "prob_proposed_before": safe_stats(prob_before),
        "prob_proposed_after": safe_stats(prob_after),
        "num_objects": len(prob_before),
        "reduction": {
            "mean": safe_stats(prob_before)["mean"] - safe_stats(prob_after)["mean"],
            "absolute_reduction": sum(b - a for b, a in zip(prob_before, prob_after)) / len(prob_before) if prob_before else 0.0,
        },
        "per_class": {},
    }
    
    for cls in classes:
        if cls in per_class_before or cls in per_class_after:
            stats["per_class"][cls] = {
                "before": safe_stats(per_class_before.get(cls, [])),
                "after": safe_stats(per_class_after.get(cls, [])),
            }
    
    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print statistics to console."""
    print("\n" + "=" * 60)
    print("POST-PROCESSING STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal objects processed: {stats['num_objects']}")
    
    print("\nP(proposed_class) BEFORE post-processing:")
    before = stats["prob_proposed_before"]
    print(f"  Mean:   {before['mean']:.4f}")
    print(f"  Median: {before['median']:.4f}")
    print(f"  Std:    {before['std']:.4f}")
    
    print("\nP(proposed_class) AFTER post-processing:")
    after = stats["prob_proposed_after"]
    print(f"  Mean:   {after['mean']:.4f}")
    print(f"  Median: {after['median']:.4f}")
    print(f"  Std:    {after['std']:.4f}")
    
    print("\nReduction in P(proposed_class):")
    reduction = stats["reduction"]
    print(f"  Mean reduction: {reduction['mean']:.4f}")
    print(f"  (Positive = proposed class probability decreased)")
    
    print("\nPer-class breakdown:")
    for cls, cls_stats in stats.get("per_class", {}).items():
        if cls_stats["before"]["count"] == 0:
            continue
        print(f"  {cls}:")
        print(f"    Before: {cls_stats['before']['mean']:.4f} (n={cls_stats['before']['count']})")
        print(f"    After:  {cls_stats['after']['mean']:.4f}")
        diff = cls_stats['before']['mean'] - cls_stats['after']['mean']
        print(f"    Change: {diff:+.4f}")


def plot_before_after_histogram(
    data: Dict[str, Any],
    classes: List[str],
    output_path: str,
) -> None:
    """
    Plot side-by-side histograms of P(proposed_class) before and after post-processing.
    """
    from matplotlib import pyplot as plt
    
    prob_before = []
    prob_after = []
    
    objects = data.get("objects", {})
    
    for img_name, obj_list in objects.items():
        for obj in obj_list:
            soft_label = obj.get("soft_label", [])
            postprocessed = obj.get("postprocessed_soft_label", [])
            proposed = obj.get("proposed_class")
                        
            if proposed is None or proposed not in classes:
                continue
            
            idx = classes.index(proposed)
            if idx < len(soft_label):
                prob_before.append(soft_label[idx])
            if idx < len(postprocessed):
                prob_after.append(postprocessed[idx])
    
    if not prob_before or not prob_after:
        print("Warning: No data to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Before histogram
    axes[0].hist(prob_before, bins=25, alpha=0.7, color="orange", edgecolor="black")
    axes[0].set_title("BEFORE: P(proposed class)")
    axes[0].set_xlabel("Probability")
    axes[0].set_ylabel("Count")
    axes[0].axvline(x=sum(prob_before)/len(prob_before), color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    # After histogram
    axes[1].hist(prob_after, bins=25, alpha=0.7, color="skyblue", edgecolor="black")
    axes[1].set_title("AFTER: P(proposed class)")
    axes[1].set_xlabel("Probability")
    axes[1].set_ylabel("Count")
    axes[1].axvline(x=sum(prob_after)/len(prob_after), color='red', linestyle='--', label='Mean')
    axes[1].legend()
    
    # Difference histogram
    diffs = [b - a for b, a in zip(prob_before, prob_after)]
    axes[2].hist(diffs, bins=25, alpha=0.7, color="purple", edgecolor="black")
    axes[2].set_title("Difference: before - after")
    axes[2].set_xlabel("Probability Change")
    axes[2].set_ylabel("Count")
    axes[2].axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    axes[2].axvline(x=sum(diffs)/len(diffs), color='red', linestyle='--', label='Mean')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply tuned post-processing to soft label files"
    )
    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "our_datasets")
    
    parser.add_argument(
        "--config",
        required=True,
        help="Config file path or shortcut (e.g., 'cityscapes' -> configs/cityscapes.json)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input soft label JSON file (path or name to search in data_dir)"
    )
    parser.add_argument(
        "--data_dir",
        default=default_data_dir,
        help="Directory containing soft label JSON files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output JSON file (default: outputs/application/<input>_postprocessed.json)"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file in place (instead of saving to outputs/application/)"
    )
    parser.add_argument(
        "--stats_output",
        default=None,
        help="Path for statistics JSON output (default: outputs/application/<input>_stats.json)"
    )
    parser.add_argument(
        "--plot_output",
        default=None,
        help="Path for before/after histogram plot (default: outputs/application/<input>_histogram.png)"
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip generating histogram plot"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Resolve config path from shortcut
    config_path = resolve_config_path(args.config)
    if not os.path.isfile(config_path):
        raise SystemExit(
            f"Config file not found: {config_path}\n"
            f"Provide a valid path or shortcut like 'cityscapes' (-> configs/cityscapes.json)"
        )
    
    # Resolve input path using find_dataset_path
    input_name = args.input[:-5] if args.input.endswith(".json") else args.input
    if os.path.isfile(args.input):
        input_path = args.input
    else:
        try:
            input_path = find_dataset_path(args.data_dir, input_name)
        except FileNotFoundError:
            available = get_available_datasets(args.data_dir)
            raise SystemExit(
                f"Input file not found: {args.input}\n"
                f"Searched in: {args.data_dir}\n"
                f"Available datasets: {', '.join(sorted(available)[:20])}"
                + (f" ... and {len(available) - 20} more" if len(available) > 20 else "")
            )
    
    # Determine output paths
    # Default output directory for all generated files
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "application")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name from input for naming output files
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    
    if args.inplace:
        # Overwrite input file
        output_path = input_path
    elif args.output:
        # User-specified output path
        output_path = args.output
    else:
        # Default: save to outputs/application/<input_name>_postprocessed.json
        output_path = os.path.join(output_dir, f"{input_basename}_postprocessed.json")
    
    # Auxiliary files always go to outputs/application/
    stats_output = args.stats_output
    if stats_output is None:
        stats_output = os.path.join(output_dir, f"{input_basename}_stats.json")
    
    plot_output = args.plot_output
    if plot_output is None:
        plot_output = os.path.join(output_dir, f"{input_basename}_histogram.png")
    
    # Load config
    config = load_config(config_path)
    multi_config = config if is_multi_config(config) else None
    
    if multi_config:
        # For multi-config, use default as the base config
        base_config = config.get("default", list(config.get("super_category_configs", {}).values())[0])
        print(f"Loaded multi-config with {len(config.get('super_category_configs', {}))} super-category configs")
    else:
        base_config = config
        print(f"Loaded config: {config.get('dataset', 'unknown')}")
        print(f"  Method: {config.get('parameters', {}).get('method', 'unknown')}")
        print(f"  Classes: {len(config.get('classes', []))}")
    
    # Validate config
    warnings = validate_config(base_config)
    if warnings:
        print("Config warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    # Load input file
    print(f"\nLoading input: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    classes = data.get("classes", base_config.get("classes", []))
    if not classes:
        raise SystemExit("No classes found in input file or config")
    
    print(f"  Dataset: {data.get('dataset_name', 'unknown')}")
    print(f"  Classes: {classes}")
    
    # Process all annotations
    objects = data.get("objects", {})
    total_objects = sum(len(obj_list) for obj_list in objects.values())
    print(f"\nProcessing {total_objects} annotations...")
    
    processed_count = 0
    merged_count = 0
    skipped_count = 0
    method_usage: Dict[str, int] = {}
    
    for img_name, obj_list in objects.items():
        for obj in obj_list:
            postprocessed_label, selected_label, prop_used, method_names = process_annotation(
                obj, base_config, multi_config
            )
            
            obj["postprocessed_soft_label"] = postprocessed_label
            obj["postprocessed_selected_soft_label"] = selected_label
            
            if prop_used is None:
                skipped_count += 1
            elif prop_used.startswith("merged_"):
                merged_count += 1
                processed_count += 1
                if method_names:
                    for method_name in method_names:
                        method_usage[method_name] = method_usage.get(method_name, 0) + 1
            else:
                processed_count += 1
                if method_names:
                    for method_name in method_names:
                        method_usage[method_name] = method_usage.get(method_name, 0) + 1
    
    print(f"  Processed: {processed_count}")
    print(f"  Merged annotations: {merged_count}")
    print(f"  Skipped (no proposed class): {skipped_count}")
    
    # Print method usage summary
    if method_usage:
        print(f"\nMethod usage:")
        for method_name, count in sorted(method_usage.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / processed_count if processed_count > 0 else 0
            print(f"  {method_name}: {count} ({pct:.1f}%)")
    
    # Compute and print statistics
    stats = compute_statistics(data, classes)
    if not args.quiet:
        print_statistics(stats)
    
    # Save output file
    print(f"\nSaving output: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    # Save statistics
    print(f"Saving statistics: {stats_output}")
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    # Generate plot
    if not args.skip_plot:
        try:
            plot_before_after_histogram(data, classes, plot_output)
        except ImportError:
            print("Warning: matplotlib not available, skipping plot")
        except Exception as e:
            print(f"Warning: Failed to generate plot: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
