#!/usr/bin/env python3
"""
Shared utilities for calibration and FP check scripts.

Contains common data loading, probability computation, and post-processing functions
used by both run_calibration.py and run_fp_check.py.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from postprocess.smoothing import counts_to_dirichlet_probs, Smoothing
from postprocess.config_loader import load_config, is_multi_config, get_config_for_class
from postprocess.methods_factory import create_method_probs
from postprocess.data_loader import DATASET_SHORTCUTS


# Derive dataset files from shortcuts (add .json extension)
DATASET_FILES = {
    k: v + ".json" for k, v in DATASET_SHORTCUTS.items()
    if not k.startswith("coco_")
}

# COCO super-category file patterns (derived from shortcuts)
COCO_SUPER_CATEGORIES = {
    v + ".json": k.replace("coco_", "")
    for k, v in DATASET_SHORTCUTS.items()
    if k.startswith("coco_")
}

# Aggregated result files (one per dataset, under our_datasets/)
AGGREGATED_FILES = {
    "cityscapes": "soft_Cityscapes_val.json",
    "pascalvoc": "soft_PascalVOC_2012_segmentation_val.json",
    "kitti": "soft_Kitti_2D_train.json",
    "coco": "soft_COCO_2017_val.json",
}


@dataclass
class ObjectCalibrationData:
    """
    Extracted calibration data for a single object.
    
    Contains all probability distributions and review data needed for
    both calibration visualization and FP checking.
    """
    obj_id: str  # Unique identifier (img_id_obj_idx)
    drawn_class: str  # The class being evaluated
    review_results: List[float]  # Individual reviewer scores (0-1)
    avg_review_score: float  # Mean of review_results
    
    # Probability for drawn_class from different distributions
    biased_prob: float  # From biased frequencies (soft_label)
    unbiased_prob: float  # From unbiased frequencies
    dirichlet_prob: float  # Dirichlet-smoothed unbiased
    postprocessed_prob: Optional[float]  # Post-processed (if config available)
    
    # Majority classes from each distribution
    biased_majority_class: str  # argmax of biased distribution
    unbiased_majority_class: str  # argmax of unbiased distribution
    dirichlet_majority_class: str  # argmax of dirichlet distribution
    postprocessed_majority_class: Optional[str]  # argmax of postprocessed (if available)
    postprocessed_selected_prob: Optional[float]  # Post-processed selected (if available)
    postprocessed_selected_majority_class: Optional[str]  # argmax of selected (if available)
    
    # Additional metadata
    proposed_class: str  # Original proposed class from object
    color_label: str  # For coloring: drawn_class_name or super-category


def load_single_dataset(data_dir: str, dataset_name: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load a single dataset file.
    
    Args:
        data_dir: Directory containing dataset files
        dataset_name: Name of dataset (e.g., 'pascalvoc', 'cityscapes')
    
    Returns:
        (data dict, classes list)
    """
    filename = DATASET_FILES.get(dataset_name.lower())
    if not filename:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_FILES.keys())}")
    
    filepath = os.path.join(data_dir, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data, data.get("classes", [])


def load_coco_datasets(data_dir: str) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """
    Load and merge all COCO super-category files.
    
    Args:
        data_dir: Directory containing COCO files
    
    Returns:
        (merged data, combined classes, obj_id -> super_category mapping)
    """
    merged_objects = {}
    all_classes = set()
    obj_to_super_category = {}
    
    for filename, super_cat in COCO_SUPER_CATEGORIES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            print(f"  Warning: COCO file not found, skipping: {filename}")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        classes = data.get("classes", [])
        all_classes.update(classes)
        
        objects = data.get("objects", {})
        for img_id, obj_list in objects.items():
            # Create unique keys by prepending super-category
            unique_img_id = f"{super_cat}_{img_id}"
            merged_objects[unique_img_id] = obj_list
            
            # Track super-category for each object
            for idx, obj in enumerate(obj_list):
                obj_id = f"{unique_img_id}_{idx}"
                obj_to_super_category[obj_id] = super_cat
    
    merged_data = {
        "objects": merged_objects,
        "classes": sorted(all_classes),
    }
    
    return merged_data, sorted(all_classes), obj_to_super_category


def load_dataset(
    data_dir: str,
    dataset_name: str,
) -> Tuple[Dict[str, Any], List[str], bool, Optional[Dict[str, str]]]:
    """
    Unified dataset loading for both single datasets and COCO.
    
    Args:
        data_dir: Directory containing dataset files
        dataset_name: Dataset name ('pascalvoc', 'cityscapes', 'kitti', 'coco')
    
    Returns:
        (data dict, classes list, is_coco flag, obj_to_super_category mapping or None)
    """
    is_coco = dataset_name.lower() == "coco"
    obj_to_super_category = None
    
    if is_coco:
        data, classes, obj_to_super_category = load_coco_datasets(data_dir)
    else:
        data, classes = load_single_dataset(data_dir, dataset_name)
    
    return data, classes, is_coco, obj_to_super_category


def load_default_config(
    dataset_name: str,
    base_dir: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load default config for a dataset.
    
    Args:
        dataset_name: Dataset name
        base_dir: Base directory (default: postprocess/../)
    
    Returns:
        (config, multi_config) - multi_config is set if config has super_category_configs
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..")
    
    config_path = os.path.join(base_dir, "configs", dataset_name + ".json")
    
    config = None
    multi_config = None
    
    try:
        config = load_config(config_path)
        if is_multi_config(config):
            multi_config = config
            config = config.get("default", list(config.get("super_category_configs", {}).values())[0])
    except Exception:
        pass
    
    return config, multi_config


def compute_postprocessed_distribution(
    obj: Dict[str, Any],
    classes: List[str],
    config: Dict[str, Any],
    multi_config: Optional[Dict[str, Any]] = None,
    target_class: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Compute post-processed probability distribution for an object.
    
    Args:
        obj: Object dict with 'frequencies' and 'proposed_class'
        classes: List of class names
        config: Default config dict
        multi_config: Multi-config dict (for COCO super-categories)
        target_class: If provided, use this class to select config (for multi-config)
    
    Returns:
        Dict mapping class names to probabilities, or None on failure
    """
    try:
        # Get appropriate config
        if multi_config and target_class:
            use_config = get_config_for_class(target_class, multi_config) or config
        else:
            use_config = config
        
        params = use_config.get("parameters", {})
        transition_matrix = use_config.get("transition_matrix", {})
        
        # Get frequencies
        frequencies = obj.get("frequencies", {})
        proposed_class = obj.get("proposed_class")
        
        if not frequencies or not proposed_class:
            return None
        
        # Create method
        method_type = params.get("method", "cleverlabel")
        smoothing_type = Smoothing[params.get("smoothing", "DIRICHLET_POST")]
        alpha = params.get("alpha", 0.5)
        epsilon = params.get("epsilon", 1e-10)
        
        method_params = {k: v for k, v in params.items()
                         if k not in ["method", "smoothing", "alpha", "epsilon"]}
        
        obj_id = "temp"
        map_id_proposed = {obj_id: proposed_class}
        
        method = create_method_probs(
            method_type=method_type,
            params=method_params,
            smoothing=smoothing_type,
            smoothing_params={"alpha": alpha, "epsilon": epsilon},
            map_id_proposed_class=map_id_proposed,
            classes=classes,
            transition_c=transition_matrix,
        )
        
        # Apply method
        result = method({obj_id: frequencies})
        return result.get(obj_id, None)
    
    except Exception as e:
        print(f"Warning: Failed to compute post-processed distribution: {e}")
        return None


def compute_postprocessed_prob(
    obj: Dict[str, Any],
    drawn_class: str,
    classes: List[str],
    config: Dict[str, Any],
    multi_config: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Compute post-processed probability for a specific class.
    
    Args:
        obj: Object dict with 'frequencies' and 'proposed_class'
        drawn_class: Class to get probability for
        classes: List of class names
        config: Default config dict
        multi_config: Multi-config dict (for COCO super-categories)
    
    Returns:
        Probability for drawn_class, or None on failure
    """
    dist = compute_postprocessed_distribution(obj, classes, config, multi_config, drawn_class)
    if dist is None:
        return None
    return dist.get(drawn_class, None)


def get_majority_class(distribution: Dict[str, float]) -> str:
    """Get the class with highest probability from a distribution."""
    if not distribution:
        return ""
    return max(distribution.keys(), key=lambda k: distribution[k])


def extract_object_calibration_data(
    data: Dict[str, Any],
    classes: List[str],
    alpha: float,
    is_coco: bool = False,
    obj_to_super_category: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    multi_config: Optional[Dict[str, Any]] = None,
    count_multiplier_cant_solves: float = 1.0,
    aggregated_index: Optional[Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]]] = None,
) -> List[ObjectCalibrationData]:
    """
    Extract calibration data for all objects with human_calibration field.
    
    This is the main shared extraction function used by both calibration
    visualization and FP checking.
    
    Args:
        data: Dataset dict with 'objects' key
        classes: List of class names
        alpha: Dirichlet smoothing alpha
        is_coco: Whether this is COCO dataset
        obj_to_super_category: Mapping from obj_id to super-category (for COCO)
        config: Post-processing config (optional)
        multi_config: Multi-config for COCO (optional)
        count_multiplier_cant_solves: Multiplier for cantsolve counts
        aggregated_index: Optional mo_id -> (biased_dist, postprocessed_dist,
            selected_dist) from aggregated files. When provided, biased_prob,
            postprocessed_prob, and postprocessed_selected_prob are taken from
            the aggregated file instead of computed on-the-fly.
            Unbiased distributions always come from unbiased_samples.
    
    Returns:
        List of ObjectCalibrationData for each object with human_calibration
    """
    results = []
    objects = data.get("objects", {})
    
    for img_id, obj_list in objects.items():
        for obj_idx, obj in enumerate(obj_list):
            human_cal = obj.get("human_calibration")
            if not human_cal:
                continue
            
            drawn_class = human_cal.get("drawn_class_name")
            review_results = human_cal.get("review_results", [])
            
            if not drawn_class or not review_results:
                continue
            
            obj_id = f"{img_id}_{obj_idx}"
            mo_id = obj.get("mo_id", "")
            
            # Check if we have aggregated data for this object
            agg_entry = None
            if aggregated_index and mo_id:
                agg_entry = aggregated_index.get(mo_id)
                if agg_entry is None:
                    print(f"Error: mo_id {mo_id} ({obj_id}) not found in aggregated index, skip")
                    continue
            
            # Compute biased distribution
            if agg_entry:
                biased_dist = agg_entry[0]
            else:
                biased_freqs = obj.get("frequencies", {})
                biased_dist = counts_to_dirichlet_probs(
                    biased_freqs, alpha=0, support=classes
                )
            biased_prob = biased_dist.get(drawn_class, 0.0)
            biased_majority = get_majority_class(biased_dist)
            
            # Compute unbiased distributions (always from unbiased_samples)
            unbiased_freqs = obj.get("unbiased_frequencies", {})
            if not unbiased_freqs:
                print(f"Warning: Object {obj_id} has no unbiased frequencies, skipping")
                continue
            
            # Apply cantsolve multiplier
            if 'cantsolve' in unbiased_freqs:
                unbiased_freqs = unbiased_freqs.copy()
                unbiased_freqs['cantsolve'] = unbiased_freqs['cantsolve'] * count_multiplier_cant_solves
            
            # Unbiased MLE distribution
            unbiased_dist = counts_to_dirichlet_probs(
                unbiased_freqs, alpha=0, support=classes
            )
            unbiased_prob = unbiased_dist.get(drawn_class, 0.0)
            unbiased_majority = get_majority_class(unbiased_dist)
            
            # Dirichlet-smoothed unbiased distribution
            dirichlet_dist = counts_to_dirichlet_probs(
                unbiased_freqs, alpha=alpha, support=classes
            )
            dirichlet_prob = dirichlet_dist.get(drawn_class, 0.0)
            dirichlet_majority = get_majority_class(dirichlet_dist)
            
            # Post-processed distribution
            postprocessed_prob = None
            postprocessed_majority = None
            if agg_entry:
                post_dist = agg_entry[1]
                if post_dist:
                    postprocessed_prob = post_dist.get(drawn_class, None)
                    postprocessed_majority = get_majority_class(post_dist)
            elif config:
                postprocessed_dist = compute_postprocessed_distribution(
                    obj, classes, config, multi_config, drawn_class
                )
                if postprocessed_dist:
                    postprocessed_prob = postprocessed_dist.get(drawn_class, None)
                    postprocessed_majority = get_majority_class(postprocessed_dist)
            
            # Post-processed selected distribution
            postprocessed_selected_prob = None
            postprocessed_selected_majority = None
            if agg_entry and len(agg_entry) > 2:
                selected_dist = agg_entry[2]
                if selected_dist:
                    postprocessed_selected_prob = selected_dist.get(drawn_class, None)
                    postprocessed_selected_majority = get_majority_class(selected_dist)
            elif config:
                # For non-aggregated source: no merging, so selected = postprocessed
                postprocessed_selected_prob = postprocessed_prob
                postprocessed_selected_majority = postprocessed_majority
            
            # Determine color label
            if is_coco and obj_to_super_category:
                color_label = obj_to_super_category.get(obj_id, "unknown")
            else:
                color_label = drawn_class
            
            # Get proposed class
            proposed_class = obj.get("proposed_class", "")
            
            # Compute average review score
            avg_review_score = sum(review_results) / len(review_results)
            
            results.append(ObjectCalibrationData(
                obj_id=obj_id,
                drawn_class=drawn_class,
                review_results=review_results,
                avg_review_score=avg_review_score,
                biased_prob=biased_prob,
                unbiased_prob=unbiased_prob,
                dirichlet_prob=dirichlet_prob,
                postprocessed_prob=postprocessed_prob,
                biased_majority_class=biased_majority,
                unbiased_majority_class=unbiased_majority,
                dirichlet_majority_class=dirichlet_majority,
                postprocessed_majority_class=postprocessed_majority,
                postprocessed_selected_prob=postprocessed_selected_prob,
                postprocessed_selected_majority_class=postprocessed_selected_majority,
                proposed_class=proposed_class,
                color_label=color_label,
            ))
    
    return results


def classify_review_bucket(
    score: float,
    thresholds: Tuple[float, float] = (0.3, 0.7),
) -> str:
    """
    Classify review score into bucket.
    
    Args:
        score: Review score (0-1)
        thresholds: (no_threshold, clearly_threshold)
            - score <= no_threshold -> "no"
            - score >= clearly_threshold -> "clearly"
            - otherwise -> "maybe"
    
    Returns:
        Bucket name: "no", "maybe", or "clearly"
    """
    no_thresh, clearly_thresh = thresholds
    if score <= no_thresh:
        return "no"
    elif score >= clearly_thresh:
        return "clearly"
    else:
        return "maybe"


def get_default_data_dir() -> str:
    """Get default data directory for unbiased_samples."""
    return os.path.join(os.path.dirname(__file__), "..", "our_datasets", "unbiased_samples")


def get_default_aggregated_dir() -> str:
    """Get default data directory for aggregated files (our_datasets/)."""
    return os.path.join(os.path.dirname(__file__), "..", "our_datasets")


def load_aggregated_index(
    aggregated_dir: str,
    dataset_name: str,
) -> Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]]:
    """
    Load aggregated file and build mo_id -> (biased_dist, postprocessed_dist, selected_dist) index.

    For merged objects (with mo_id_2), both mo_id and mo_id_2 are indexed.

    Args:
        aggregated_dir: Directory containing soft_*.json files
        dataset_name: Dataset name (cityscapes, pascalvoc, kitti, coco)

    Returns:
        Dict mapping mo_id -> (biased_dist_dict, postprocessed_dist_dict, selected_dist_dict)
        where each dist_dict maps class_name -> probability
    """
    filename = AGGREGATED_FILES.get(dataset_name.lower())
    if not filename:
        raise ValueError(f"No aggregated file for dataset: {dataset_name}")

    filepath = os.path.join(aggregated_dir, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Aggregated file not found: {filepath}")

    print(f"  Loading aggregated file: {filename}")
    with open(filepath, "r", encoding="utf-8") as f:
        agg_data = json.load(f)

    classes = agg_data.get("classes", [])
    if not classes:
        raise ValueError(f"No classes found in aggregated file: {filepath}")

    index: Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]] = {}
    n_merged = 0

    for img_id, obj_list in agg_data.get("objects", {}).items():
        for obj in obj_list:
            mo_id = obj.get("mo_id")
            if not mo_id:
                continue

            # Convert soft_label list -> {class: prob} dict
            soft_label = obj.get("soft_label", [])
            biased_dist = dict(zip(classes, soft_label)) if len(soft_label) == len(classes) else {}

            # Convert postprocessed_soft_label list -> dict
            post_label = obj.get("postprocessed_soft_label", [])
            post_dist = dict(zip(classes, post_label)) if len(post_label) == len(classes) else {}

            # Convert postprocessed_selected_soft_label list -> dict
            selected_label = obj.get("postprocessed_selected_soft_label", [])
            selected_dist = dict(zip(classes, selected_label)) if len(selected_label) == len(classes) else {}

            entry = (biased_dist, post_dist, selected_dist)
            index[mo_id] = entry

            # Also index by mo_id_2 for merged objects
            mo_id_2 = obj.get("mo_id_2")
            if mo_id_2:
                index[mo_id_2] = entry
                n_merged += 1

    print(f"  Aggregated index: {len(index)} mo_ids ({n_merged} from merged objects)")
    return index
