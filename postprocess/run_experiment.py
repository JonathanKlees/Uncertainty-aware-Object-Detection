#!/usr/bin/env python3
"""
Main experiment runner for post-processing biased annotation distributions.

Usage:
    # Use shorthand dataset names:
    python -m postprocess.run_experiment --dataset cityscapes
    python -m postprocess.run_experiment --dataset coco_kitchen
    
    # Or full dataset name:
    python -m postprocess.run_experiment --dataset Cityscapes_sample_with_unbiased_annotations
    
    # Use legacy evaluation:
    python -m postprocess.run_experiment --dataset kitti --legacy
"""

import argparse
import os

from postprocess.clever_labeling import get_tranistion_matrix
from postprocess.data_loader import (
    apply_cantsolve_multiplier,
    load_dataset_from_json,
    load_annotation_timing,
    filter_to_common_keys,
    create_distributions,
    get_available_datasets,
    find_dataset_path,
    resolve_dataset_name,
    DATASET_SHORTCUTS,
)
from postprocess.evaluation import (
    evaluate_method,
    print_evaluation_results,
    print_baseline_comparison,
    print_majority_changes,
    compute_speedup_metrics,
    compute_damper_values,
    compute_per_class_dampers,
    print_speedup_metrics,
    print_damper_results,
)
from postprocess.evaluation_legacy import (
    evaluate_postprocess_method_legacy,
)
from postprocess.smoothing import Smoothing, smooth_counts
from postprocess.metrics import Metric
from postprocess.methods_factory import get_default_methods
from postprocess.methods_legacy import get_default_methods_legacy
from postprocess.visualization import (
    plot_probability_histograms,
    plot_probability_difference,
    plot_annotation_times,
    plot_damper_histogram,
    plot_damper_per_class,
    print_transition_matrix,
    print_filtered_examples,
    build_comparison_dataframe,
)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process unbiased vs biased distributions from JSON dataset files"
    )
    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "our_datasets", "unbiased_samples")
    
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
        "--output_dir", 
        default="outputs/experiments",
        help="Directory for output files (default: outputs/experiments)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dirichlet smoothing parameter for ground truth and post-processing"
    )
    parser.add_argument(
        "--metric_bootstrapping",
        type=str,
        choices=["kl", "js", "js_sqrt"],
        default="js_sqrt",
        help="Divergence metric for bootstrapping: kl, js, or js_sqrt"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy evaluation (counts-based, no smoothing in methods)"
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip generating plot files"
    )
    parser.add_argument(
        "--sample_plots",
        action="store_true",
        help="Generate sample distribution bar charts (best/worst/random)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples per category for sample plots (default: 10)"
    )
    parser.add_argument(
        "--analyze_annotations",
        action="store_true",
        help="Analyze annotation timing (speedup) and damper values"
    )
    
    args = parser.parse_args()

    # Resolve shorthand dataset name and find path
    dataset_short = args.dataset
    dataset_name = resolve_dataset_name(dataset_short)
    try:
        dataset_path = find_dataset_path(args.data_dir, dataset_name)
    except FileNotFoundError:
        available = get_available_datasets(args.data_dir)
        raise SystemExit(
            f"Dataset not found: {args.dataset}\n"
            f"Available datasets: {', '.join(sorted(available))}"
        )

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class, classes = (
        load_dataset_from_json(dataset_path)
    )

    print(f"Loaded dataset '{dataset_name}':")
    print(f"  Objects: {len(map_id_biased_counts)}")
    print(f"  Classes: {classes}")

    # Filter to common keys
    map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class = filter_to_common_keys(
        map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class
    )
    map_id_unbiased_counts = apply_cantsolve_multiplier(map_id_unbiased_counts) 
    # map_id_biased_counts = apply_cantsolve_multiplier(map_id_biased_counts) 
    print(f"  Objects with both biased and unbiased annotations: {len(map_id_biased_counts)}")

    # Create distributions    
    map_id_unbiased_dist = create_distributions(map_id_unbiased_counts)
    map_id_biased_dist = create_distributions(map_id_biased_counts)

    # Build comparison DataFrame
    df = build_comparison_dataframe(
        map_id_unbiased_dist, 
        map_id_biased_dist, 
        map_id_proposed_class
    )

    # Generate plots
    if not args.skip_plots:
        unbiased_probs = df["unbiased"].tolist()
        biased_probs = df["biased"].tolist()

        plot_probability_histograms(
            unbiased_probs, 
            biased_probs,
            os.path.join(output_dir, f"{dataset_short}_hist_unbiased_vs_biased.png")
        )
        plot_probability_difference(
            unbiased_probs, 
            biased_probs,
            os.path.join(output_dir, f"{dataset_short}_hist_diff_biased_minus_unbiased.png")
        )

    # Print filtered examples
    print_filtered_examples(df, threshold=0.3)

    # Count majority changes
    print_majority_changes(map_id_unbiased_dist, map_id_biased_dist)

    # Print baseline comparison
    unbiased_smoothing = Smoothing.EPSILON
    baseline_summary = print_baseline_comparison(map_id_unbiased_counts, map_id_biased_counts, classes, alpha=args.alpha, smoothing=unbiased_smoothing)

    # Calculate transition matrix
    transition_c = get_tranistion_matrix(classes, map_id_unbiased_counts)
    print_transition_matrix(classes, transition_c)

    # Calculate per-class dampers from training data (full dataset as training for run_experiment)
    per_class_dampers = compute_per_class_dampers(
        map_id_biased_dist, map_id_unbiased_dist, map_id_proposed_class, classes
    )
    print(f"\nPer-class dampers (median): {per_class_dampers}")

    # Annotation analysis (speedup and damper)
    if args.analyze_annotations:
        # Load timing data
        timing_data = load_annotation_timing(dataset_path)
        print(f"\nLoaded annotation timing for {len(timing_data)} objects")
        
        # Compute and print speedup metrics
        speedup = compute_speedup_metrics(timing_data)
        print_speedup_metrics(speedup)
        
        # Compute and print damper values
        damper_result = compute_damper_values(
            map_id_biased_dist, map_id_unbiased_dist, map_id_proposed_class
        )
        print_damper_results(damper_result)
        
        # Generate plots if not skipped
        if not args.skip_plots:
            # Collect all times
            all_biased_times = []
            all_unbiased_times = []
            for timing in timing_data.values():
                all_biased_times.extend(timing.biased_times)
                all_unbiased_times.extend(timing.unbiased_times)
            
            # Plot annotation times
            if all_biased_times and all_unbiased_times:
                plot_annotation_times(
                    all_biased_times,
                    all_unbiased_times,
                    os.path.join(output_dir, f"{dataset_short}_annotation_times.png")
                )
            
            # Plot damper histogram (global)
            if damper_result.global_dampers:
                plot_damper_histogram(
                    damper_result.global_dampers,
                    os.path.join(output_dir, f"{dataset_short}_damper_histogram.png"),
                    max_damper=2.0
                )
            
            # Plot per-class damper
            if damper_result.per_class_dampers:
                plot_damper_per_class(
                    damper_result.per_class_dampers,
                    os.path.join(output_dir, f"{dataset_short}_damper_per_class.png"),
                    max_damper=2.0
                )

    # Parse metric option
    metric_map = {
        "kl": Metric.KL,
        "js": Metric.JS,
        "js_sqrt": Metric.JS_SQRT,
    }
    metric_bootstrapping = metric_map[args.metric_bootstrapping]
    
    # Get baseline JS_SQRT for percentage calculations
    baseline_js_sqrt = baseline_summary.get("js_sqrt", {}).get("mean", 0.0)

    if args.legacy:
        # Legacy path: methods return counts, legacy evaluation
        methods = get_default_methods_legacy(
            map_id_proposed_class=map_id_proposed_class,
            classes=classes,
            transition_c=transition_c,
            map_id_unbiased_counts=map_id_unbiased_counts,
        )

        for name, method in methods:
            post_processed_counts = method(map_id_biased_counts)
            results = evaluate_postprocess_method_legacy(
                map_id_unbiased_counts,
                map_id_biased_counts,
                post_processed_counts,
                map_id_proposed_class,
                alpha=args.alpha,
            )
            print_evaluation_results(name, results, baseline_js_sqrt)
    else:
        # New path: methods return probs with all smoothing variants
        methods = get_default_methods(
            map_id_proposed_class=map_id_proposed_class,
            classes=classes,
            transition_c=transition_c,
            map_id_unbiased_counts=map_id_unbiased_counts,
            per_class_dampers=per_class_dampers,
        )
        
        # Pre-compute ground truth probs for each smoothing variant
        from postprocess.methods_factory import DEFAULT_SMOOTHING_VARIANTS
        ground_truth_cache = {}
        for smoothing_type, smoothing_params, suffix in DEFAULT_SMOOTHING_VARIANTS:
            alpha = smoothing_params.get("alpha", args.alpha)
            epsilon = smoothing_params.get("epsilon", 1e-10)

            # TODO always used baseline smoothing as baseline
            smoothing_type = unbiased_smoothing
            epsilon = 1e-10

            ground_truth_cache[suffix] = {
                "unbiased": {
                    k: smooth_counts(c, smoothing_type, classes, alpha, epsilon)
                    for k, c in map_id_unbiased_counts.items()
                },
                "biased": {
                    k: smooth_counts(c, smoothing_type, classes, alpha, epsilon)
                    for k, c in map_id_biased_counts.items()
                },
            }

        for name, method in methods:
            post_processed_probs = method(map_id_biased_counts)
            
            # Determine smoothing type from method name suffix
            smoothing_suffix = None
            for suffix in ground_truth_cache.keys():
                if name.endswith(suffix):
                    smoothing_suffix = suffix
                    break
            
            if smoothing_suffix is None:
                print(f"Warning: Could not determine smoothing for method '{name}', skipping")
                continue
            
            unbiased_probs = ground_truth_cache[smoothing_suffix]["unbiased"]
            biased_probs = ground_truth_cache[smoothing_suffix]["biased"]
            
            # Prepare sample plot path if requested
            sample_plot_path = None
            if args.sample_plots:
                safe_name = name.replace("/", "_").replace(" ", "_")
                sample_plot_path = os.path.join(
                    output_dir, f"{dataset_short}_sample_dists_{safe_name}.png"
                )
            
            results = evaluate_method(
                unbiased_probs,
                biased_probs,
                post_processed_probs,
                map_id_proposed_class,
                metric=metric_bootstrapping,
                sample_plot_path=sample_plot_path,
                num_samples=args.num_samples,
            )
            print_evaluation_results(name, results, baseline_js_sqrt)


if __name__ == "__main__":
    main()
