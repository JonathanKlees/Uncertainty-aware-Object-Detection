"""
Data loading utilities for loading datasets from JSON files in our_datasets/.

Handles:
- Loading biased/unbiased frequency counts
- Extracting proposed classes
- Converting counts to distributions
- Filtering to common keys
- Splitting into train/validation sets
- Dataset name shortcuts
"""

import json
import os
import random
from typing import Dict, List, Tuple

from postprocess.smoothing import Counts, Dataset


# Mapping of shorthand names to full dataset names
DATASET_SHORTCUTS = {
    "cityscapes": "Cityscapes_sample_with_unbiased_annotations",
    "kitti": "Kitti_sample_with_unbiased_annotations",
    "pascalvoc": "PascalVOC_sample_with_unbiased_annotations",
    "coco_accessory": "COCO_Accessory_sample_with_unbiased_annotations",
    "coco_animal": "COCO_Animal_sample_with_unbiased_annotations",
    "coco_electronic": "COCO_Electronic_sample_with_unbiased_annotations",
    "coco_food": "COCO_Food_sample_with_unbiased_annotations",
    "coco_furniture": "COCO_Furniture_sample_with_unbiased_annotations",
    "coco_indoor": "COCO_Indoor_sample_with_unbiased_annotations",
    "coco_kitchen": "COCO_KitchenAppliance_sample_with_unbiased_annotations",
    "coco_outdoor": "COCO_Outdoor_sample_with_unbiased_annotations",
    "coco_person": "COCO_Person_sample_with_unbiased_annotations",
    "coco_sports": "COCO_Sports_sample_with_unbiased_annotations",
    "coco_vehicle": "COCO_Vehicle_sample_with_unbiased_annotations",
}


def resolve_dataset_name(name: str) -> str:
    """Resolve a shorthand dataset name to the full name."""
    lower_name = name.lower()
    if lower_name in DATASET_SHORTCUTS:
        return DATASET_SHORTCUTS[lower_name]
    return name  # Return as-is if not a shorthand


def get_proposed_class_from_unbiased(unbiased_frequencies: Dict[str, int]) -> str:
    """
    Dummy method to get the proposed class from unbiased frequencies.
    Takes the class with the highest unbiased frequency as the proposal.
    
    Note: This should later be loaded directly from the data files.
    """
    if not unbiased_frequencies:
        return "unknown"
    return max(unbiased_frequencies, key=unbiased_frequencies.get)


def counts_to_distribution(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert raw counts to a probability distribution."""
    total = sum(counts.values())
    return {cls: value / total for cls, value in counts.items()} if total > 0 else {}


def apply_cantsolve_multiplier(
    map_id_counts: Dataset,
    multiplier: float = 1.0,
    cantsolve_key: str = "cantsolve",
) -> Dataset:
    """
    Apply a multiplier to the cantsolve counts in a dataset.
    
    Args:
        map_id_counts: Map of object_id to frequency counts
        multiplier: Factor to multiply cantsolve counts by (default: 2.0)
        cantsolve_key: Key for cantsolve class (default: "cantsolve")
    
    Returns:
        New dataset with adjusted cantsolve counts
    """
    result = {}
    for obj_id, counts in map_id_counts.items():
        new_counts = dict(counts)  # Copy
        if cantsolve_key in new_counts:
            new_counts[cantsolve_key] = new_counts[cantsolve_key] * multiplier
        result[obj_id] = new_counts
    return result


def load_dataset_from_json(json_path: str) -> Tuple[Dataset, Dataset, Dict[str, str], List[str]]:
    """
    Load dataset from JSON file in our_datasets/.
    
    Args:
        json_path: Path to the JSON dataset file
        
    Returns:
        - map_id_biased_counts: dict[obj_id -> biased frequency counts]
        - map_id_unbiased_counts: dict[obj_id -> unbiased frequency counts]  
        - map_id_proposed_class: dict[obj_id -> proposed class]
        - classes: list of class names
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    classes = data.get("classes", [])
    objects = data.get("objects", {})
    
    map_id_biased_counts: Dataset = {}
    map_id_unbiased_counts: Dataset = {}
    map_id_proposed_class: Dict[str, str] = {}
    
    for image_id, obj_list in objects.items():
        for idx, obj in enumerate(obj_list):
            # Create unique object ID from image_id and object index
            obj_id = f"{image_id}_{idx}"
            
            # Get biased frequencies
            biased_freqs = obj.get("frequencies", {})
            if biased_freqs:
                map_id_biased_counts[obj_id] = biased_freqs
            
            # Get unbiased frequencies
            unbiased_freqs = obj.get("unbiased_frequencies", {})
            if unbiased_freqs:
                map_id_unbiased_counts[obj_id] = unbiased_freqs
            
            # Get proposed class directly from data file
            proposed_class = obj.get("proposed_class")
            if proposed_class:
                map_id_proposed_class[obj_id] = proposed_class
            elif biased_freqs:
                # Fallback: determine from biased frequencies if not in file
                map_id_proposed_class[obj_id] = get_proposed_class_from_unbiased(biased_freqs)
    
    return map_id_biased_counts, map_id_unbiased_counts, map_id_proposed_class, classes


def filter_to_common_keys(
    map_id_biased_counts: Dataset,
    map_id_unbiased_counts: Dataset,
    map_id_proposed_class: Dict[str, str],
) -> Tuple[Dataset, Dataset, Dict[str, str]]:
    """
    Filter all maps to only include keys present in all three.
    
    Returns:
        Filtered versions of (biased_counts, unbiased_counts, proposed_class)
    """
    common_keys = (
        set(map_id_unbiased_counts.keys()) 
        & set(map_id_biased_counts.keys()) 
        & set(map_id_proposed_class.keys())
    )
    
    return (
        {k: v for k, v in map_id_biased_counts.items() if k in common_keys},
        {k: v for k, v in map_id_unbiased_counts.items() if k in common_keys},
        {k: v for k, v in map_id_proposed_class.items() if k in common_keys},
    )


def create_distributions(counts_map: Dataset) -> Dict[str, Dict[str, float]]:
    """Convert a map of counts to a map of distributions."""
    return {
        obj_id: counts_to_distribution(counts) 
        for obj_id, counts in counts_map.items()
    }


def get_available_datasets(data_dir: str) -> List[str]:
    """List available dataset names in the data directory and subdirectories."""
    if not os.path.isdir(data_dir):
        return []
    
    datasets = []
    # Main directory
    datasets.extend(f[:-5] for f in os.listdir(data_dir) if f.endswith(".json"))
    
    # Check unbiased_samples subdirectory
    unbiased_dir = os.path.join(data_dir, "unbiased_samples")
    if os.path.isdir(unbiased_dir):
        datasets.extend(f[:-5] for f in os.listdir(unbiased_dir) if f.endswith(".json"))
    
    return datasets


def find_dataset_path(data_dir: str, dataset_name: str) -> str:
    """
    Find the full path to a dataset file. Searches main dir and unbiased_samples/.
    
    Args:
        data_dir: Base data directory
        dataset_name: Name of dataset (without .json extension)
        
    Returns:
        Full path to dataset file
        
    Raises:
        FileNotFoundError: If dataset not found
    """
    # Check main directory first
    main_path = os.path.join(data_dir, f"{dataset_name}.json")
    if os.path.isfile(main_path):
        return main_path
    
    # Check unbiased_samples subdirectory
    unbiased_path = os.path.join(data_dir, "unbiased_samples", f"{dataset_name}.json")
    if os.path.isfile(unbiased_path):
        return unbiased_path
    
    raise FileNotFoundError(f"Dataset not found: {dataset_name}")


def split_dataset(
    map_id_biased_counts: Dataset,
    map_id_unbiased_counts: Dataset,
    map_id_proposed_class: Dict[str, str],
    train_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[
    Tuple[Dataset, Dataset, Dict[str, str]],
    Tuple[Dataset, Dataset, Dict[str, str]],
]:
    """
    Split dataset into train and validation sets.
    
    Args:
        map_id_biased_counts: Object ID -> biased counts
        map_id_unbiased_counts: Object ID -> unbiased counts
        map_id_proposed_class: Object ID -> proposed class
        train_ratio: Fraction of data to use for training (tuning)
        seed: Random seed for reproducibility
    
    Returns:
        ((train_biased, train_unbiased, train_proposed),
         (val_biased, val_unbiased, val_proposed))
    """
    rng = random.Random(seed)
    
    # Get all keys and shuffle
    all_keys = list(map_id_biased_counts.keys())
    rng.shuffle(all_keys)
    
    # Split
    n_train = int(len(all_keys) * train_ratio)
    train_keys = set(all_keys[:n_train])
    val_keys = set(all_keys[n_train:])
    
    # Create train split
    train_biased = {k: v for k, v in map_id_biased_counts.items() if k in train_keys}
    train_unbiased = {k: v for k, v in map_id_unbiased_counts.items() if k in train_keys}
    train_proposed = {k: v for k, v in map_id_proposed_class.items() if k in train_keys}
    
    # Create validation split
    val_biased = {k: v for k, v in map_id_biased_counts.items() if k in val_keys}
    val_unbiased = {k: v for k, v in map_id_unbiased_counts.items() if k in val_keys}
    val_proposed = {k: v for k, v in map_id_proposed_class.items() if k in val_keys}
    
    return (
        (train_biased, train_unbiased, train_proposed),
        (val_biased, val_unbiased, val_proposed),
    )


# =============================================================================
# Annotation Timing Data
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class AnnotationTiming:
    """Per-object annotation timing data."""
    biased_times: List[float] = field(default_factory=list)
    biased_votes: List[str] = field(default_factory=list)
    unbiased_times: List[float] = field(default_factory=list)
    unbiased_votes: List[str] = field(default_factory=list)


def load_annotation_timing(json_path: str) -> Dict[str, AnnotationTiming]:
    """
    Load annotation timing data from JSON dataset file.
    
    Parses `annotation_info` (biased) and `unbiased_annotation_info` fields
    to extract per-annotator vote and time.
    
    Args:
        json_path: Path to the JSON dataset file
        
    Returns:
        Dict mapping object_id -> AnnotationTiming with times and votes
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    objects = data.get("objects", {})
    result: Dict[str, AnnotationTiming] = {}
    
    for image_id, obj_list in objects.items():
        for idx, obj in enumerate(obj_list):
            obj_id = f"{image_id}_{idx}"
            timing = AnnotationTiming()
            
            # Parse biased annotation_info (dict: annotator_id -> list of {vote, time})
            annotation_info = obj.get("annotation_info", {})
            for annotator_id, votes_list in annotation_info.items():
                for vote_entry in votes_list:
                    if isinstance(vote_entry, dict):
                        vote = vote_entry.get("vote")
                        time_val = vote_entry.get("time")
                        if vote is not None and time_val is not None:
                            timing.biased_times.append(float(time_val))
                            timing.biased_votes.append(vote)
            
            # Parse unbiased_annotation_info (list of dicts: {annotator_id: {vote, time}})
            unbiased_info = obj.get("unbiased_annotation_info", [])
            for entry in unbiased_info:
                if isinstance(entry, dict):
                    for annotator_id, vote_data in entry.items():
                        if isinstance(vote_data, dict):
                            vote = vote_data.get("vote")
                            time_val = vote_data.get("time")
                            if vote is not None and time_val is not None:
                                timing.unbiased_times.append(float(time_val))
                                timing.unbiased_votes.append(vote)
            
            # Only include objects with timing data
            if timing.biased_times or timing.unbiased_times:
                result[obj_id] = timing
    
    return result

