#!/usr/bin/env python3
"""
Run parameter tuning for all datasets and create config files.

Creates:
- configs/cityscapes.json - Single config for Cityscapes
- configs/kitti.json - Single config for Kitti  
- configs/pascalvoc.json - Single config for PascalVOC
- configs/coco.json - Multi-config with per-super-category parameters

Usage:
    python -m postprocess.run_tuning_all
    python -m postprocess.run_tuning_all --datasets cityscapes kitti
    python -m postprocess.run_tuning_all --skip_coco
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Any

from postprocess.clever_labeling import get_tranistion_matrix
from postprocess.data_loader import (
    load_dataset_from_json,
    filter_to_common_keys,
    find_dataset_path,
    split_dataset,
    DATASET_SHORTCUTS,
)
from postprocess.methods_factory import (
    DEFAULT_TUNING_GRID,
    SMOOTHING_PARAM_GRID,
    SMOOTHING_PARAM_GRID_FAST,
)
from postprocess.run_tuning import tune_on_train, extract_metric


# Derive mappings from centralized DATASET_SHORTCUTS
# Mapping from COCO dataset names to super categories
COCO_DATASET_TO_SUPER_CATEGORY = {
    v: k.replace("coco_", "")
    for k, v in DATASET_SHORTCUTS.items()
    if k.startswith("coco_")
}

# Single-dataset configs (non-COCO)
SINGLE_DATASETS = {
    k: v for k, v in DATASET_SHORTCUTS.items()
    if not k.startswith("coco_")
}


def run_tuning_for_dataset(
    dataset_name: str,
    data_dir: str,
    metric: str = "delta_summary.mean.point",
    train_ratio: float = 0.5,
    seed: int = 42,
    smoothing_grid: Optional[Dict] = None,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Run tuning for a single dataset and return the config dict.
    
    Returns:
        Config dict with parameters, transition_matrix, classes, etc.
        None if dataset doesn't exist or tuning fails.
    """
    try:
        dataset_path = find_dataset_path(data_dir, dataset_name)
    except FileNotFoundError:
        print(f"  WARNING: Dataset not found: {dataset_name}")
        return None
    
    try:
        # Load dataset
        map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class, classes = (
            load_dataset_from_json(dataset_path)
        )
        
        if verbose:
            print(f"  Loaded {len(map_id_biased_counts)} objects, {len(classes)} classes")
        
        # Filter to common keys
        map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class = filter_to_common_keys(
            map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class
        )
        
        if len(map_id_biased_counts) < 10:
            print(f"  WARNING: Too few objects ({len(map_id_biased_counts)}), skipping")
            return None
        
        # Split into train/validation
        (train_biased, train_unbiased, train_proposed), \
        (val_biased, val_unbiased, val_proposed) = split_dataset(
            map_id_biased_counts,
            map_id_unbiased_counts,
            map_id_proposed_class,
            train_ratio=train_ratio,
            seed=seed,
        )
        
        if verbose:
            print(f"  Train: {len(train_biased)}, Val: {len(val_biased)}")
        
        # Calculate transition matrix on TRAIN data only
        transition_c = get_tranistion_matrix(classes, train_unbiased)
        
        # Tune on train set
        best_params, best_score, best_name = tune_on_train(
            train_biased,
            train_unbiased,
            train_proposed,
            classes,
            transition_c,
            tuning_grid=DEFAULT_TUNING_GRID,
            smoothing_grid=smoothing_grid,
            metric=metric,
            minimize=True,
            verbose=verbose,
        )
        
        if verbose:
            print(f"  Best: {best_name} (score={best_score:.6f})")
        
        # Build config
        config = {
            "dataset": dataset_name,
            "parameters": best_params,
            "tuning_metadata": {
                "metric": metric,
                "score": best_score,
                "train_ratio": train_ratio,
                "seed": seed,
            },
            "transition_matrix": transition_c,
            "classes": classes,
            
        }
        
        return config
        
    except Exception as e:
        print(f"  ERROR tuning {dataset_name}: {e}")
        return None


def save_single_config(config: Dict[str, Any], output_path: str) -> None:
    """Save a single dataset config to JSON."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {output_path}")


def save_multi_config(
    super_category_configs: Dict[str, Dict[str, Any]],
    output_path: str,
    default_super_category: str = "person",
) -> None:
    """
    Save a multi-config for COCO with per-super-category parameters.
    
    Args:
        super_category_configs: Dict mapping super category name to config
        output_path: Path to save the multi-config
        default_super_category: Which super category to use as default fallback
    """
    # Use the specified super category as default, or first available
    if default_super_category in super_category_configs:
        default_config = super_category_configs[default_super_category]
    else:
        default_config = list(super_category_configs.values())[0]
    
    multi_config = {
        "super_category_configs": super_category_configs,
        "default": default_config,
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(multi_config, f, indent=2)
    print(f"  Saved multi-config: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter tuning for all datasets"
    )
    default_data_dir = os.path.join(os.path.dirname(__file__), "..", "our_datasets")
    default_config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    
    parser.add_argument(
        "--data_dir",
        default=default_data_dir,
        help="Directory containing dataset JSON files"
    )
    parser.add_argument(
        "--config_dir",
        default=default_config_dir,
        help="Directory to save config files"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to tune (cityscapes, kitti, pascalvoc, coco). Default: all"
    )
    parser.add_argument(
        "--skip_coco",
        action="store_true",
        help="Skip COCO super-category tuning"
    )
    parser.add_argument(
        "--metric",
        default="delta_summary.mean.point",
        help="Metric path to optimize (e.g., all_metrics.js_sqrt.mean, delta_summary.mean.point)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Train/validation split ratio"
    )
    parser.add_argument(
        "--fast_smoothing_grid",
        action="store_true",
        help="Use fast smoothing parameter grid (faster but less thorough)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    if args.datasets:
        datasets_to_process = [d.lower() for d in args.datasets]
    else:
        datasets_to_process = ["cityscapes", "kitti", "pascalvoc"]
        if not args.skip_coco:
            datasets_to_process.append("coco")
    
    os.makedirs(args.config_dir, exist_ok=True)
    
    print("=" * 60)
    print("TUNING ALL DATASETS")
    print("=" * 60)
    print(f"Datasets: {datasets_to_process}")
    print(f"Metric: {args.metric}")
    print(f"Config directory: {args.config_dir}")
    print()
    
    # Select smoothing grid
    smoothing_grid = SMOOTHING_PARAM_GRID if not args.fast_smoothing_grid else SMOOTHING_PARAM_GRID_FAST
    
    # Process single datasets (Cityscapes, Kitti, PascalVOC)
    for dataset_key in ["cityscapes", "kitti", "pascalvoc"]:
        if dataset_key not in datasets_to_process:
            continue
        
        dataset_name = SINGLE_DATASETS[dataset_key]
        print(f"\n{'='*60}")
        print(f"Tuning: {dataset_key.upper()}")
        print(f"{'='*60}")
        
        config = run_tuning_for_dataset(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            metric=args.metric,
            train_ratio=args.train_ratio,
            seed=args.seed,
            smoothing_grid=smoothing_grid,
            verbose=not args.quiet,
        )
        
        if config:
            output_path = os.path.join(args.config_dir, f"{dataset_key}.json")
            save_single_config(config, output_path)
    
    # Process COCO (multi-config with super categories)
    if "coco" in datasets_to_process:
        print(f"\n{'='*60}")
        print("Tuning: COCO (per super-category)")
        print(f"{'='*60}")
        
        super_category_configs = {}
        
        for dataset_name, super_cat in COCO_DATASET_TO_SUPER_CATEGORY.items():
            print(f"\n  --- {super_cat.upper()} ({dataset_name}) ---")
            
            config = run_tuning_for_dataset(
                dataset_name=dataset_name,
                data_dir=args.data_dir,
                metric=args.metric,
                train_ratio=args.train_ratio,
                seed=args.seed,
                smoothing_grid=smoothing_grid,
                verbose=not args.quiet,
            )
            
            if config:
                super_category_configs[super_cat] = config
        
        if super_category_configs:
            output_path = os.path.join(args.config_dir, "coco.json")
            save_multi_config(super_category_configs, output_path)
            print(f"\n  Created COCO multi-config with {len(super_category_configs)} super-categories")
        else:
            print("\n  WARNING: No COCO configs created")
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Configs saved to: {args.config_dir}/")


if __name__ == "__main__":
    main()
