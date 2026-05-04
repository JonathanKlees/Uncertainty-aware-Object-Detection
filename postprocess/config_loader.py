"""
Configuration loading utilities for post-processing.

Supports:
- Single dataset configs (Cityscapes, Pascal, Kitti)
- Multi-config for COCO with per-super-category parameters
"""

import json
import os
from typing import Any, Dict, List, Optional


# COCO 80 classes mapped to their super categories
# Based on COCO dataset standard category definitions
COCO_SUPER_CATEGORIES: Dict[str, str] = {
    # person (1 class)
    "person": "person",
    
    # vehicle (8 classes)
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "airplane": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    
    # outdoor (5 classes)
    "trafficlight": "outdoor",
    "firehydrant": "outdoor",
    "stopsign": "outdoor",
    "parkingmeter": "outdoor",
    "bench": "outdoor",
    
    # animal (10 classes)
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
    
    # accessory (5 classes)
    "backpack": "accessory",
    "umbrella": "accessory",
    "handbag": "accessory",
    "tie": "accessory",
    "suitcase": "accessory",
    
    # sports (10 classes)
    "frisbee": "sports",
    "skis": "sports",
    "snowboard": "sports",
    "sportsball": "sports",
    "kite": "sports",
    "baseballbat": "sports",
    "baseballglove": "sports",
    "skateboard": "sports",
    "surfboard": "sports",
    "tennisracket": "sports",
    
    # kitchen (7 classes)
    "bottle": "kitchen",
    "wineglass": "kitchen",
    "cup": "kitchen",
    "fork": "kitchen",
    "knife": "kitchen",
    "spoon": "kitchen",
    "bowl": "kitchen",
    
    # food (10 classes)
    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hotdog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",
    
    # furniture (6 classes)
    "chair": "furniture",
    "couch": "furniture",
    "pottedplant": "furniture",
    "bed": "furniture",
    "diningtable": "furniture",
    "toilet": "furniture",
    
    # electronic (6 classes)
    "tv": "electronic",
    "laptop": "electronic",
    "mouse": "electronic",
    "remote": "electronic",
    "keyboard": "electronic",
    "cellphone": "electronic",
    
    # appliance (5 classes) - mapped to kitchen since COCO dataset combines them
    "microwave": "kitchen",
    "oven": "kitchen",
    "toaster": "kitchen",
    "sink": "kitchen",
    "refrigerator": "kitchen",
    
    # indoor (7 classes)
    "book": "indoor",
    "clock": "indoor",
    "vase": "indoor",
    "scissors": "indoor",
    "teddybear": "indoor",
    "hairdrier": "indoor",
    "toothbrush": "indoor",
}

# List of all COCO super categories (as used in tuning datasets)
# Note: kitchen and appliance are combined in the tuning dataset
COCO_SUPER_CATEGORY_LIST = [
    "person", "vehicle", "outdoor", "animal", "accessory",
    "sports", "kitchen", "food", "furniture", "electronic", "indoor"
]


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a post-processing configuration from JSON.
    
    Automatically detects single configs vs multi-configs (COCO with per-super-category params).
    
    Args:
        path: Path to the config JSON file
        
    Returns:
        Config dict. For single configs: has keys dataset, parameters, transition_matrix, classes.
        For multi-configs: has super_category_configs and optional default.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = json.load(f)
    
    # Check if this is a multi-config
    if is_multi_config(config):
        # Multi-config - validate structure
        if not config["super_category_configs"]:
            raise ValueError("Multi-config has empty 'super_category_configs'")
        return config
    
    # Single config - validate required keys
    required_keys = ["parameters", "transition_matrix", "classes"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    
    return config


def is_multi_config(config: Dict[str, Any]) -> bool:
    """Check if a config is a multi-config (has super_category_configs key)."""
    return "super_category_configs" in config


def get_super_category(class_name: str) -> Optional[str]:
    """
    Get the COCO super category for a given class name.
    
    Args:
        class_name: Name of the class (e.g., "car", "person")
        
    Returns:
        Super category name or None if not found
    """
    super_category =  COCO_SUPER_CATEGORIES.get(class_name.lower(), None)
    if super_category is not None:
        return super_category
    # try removed spaces and special characters for matching
    cleaned_name = class_name.lower().replace(" ", "").replace("-", "")
    super_category = COCO_SUPER_CATEGORIES.get(cleaned_name, None)
    return super_category



def get_config_for_class(
    class_name: str,
    multi_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Get the appropriate config for a class from a multi-config.
    
    Args:
        class_name: Name of the proposed class
        multi_config: Multi-config dict with super_category_configs
        
    Returns:
        Config dict for the appropriate super category, or default if not found
    """
    super_cat = get_super_category(class_name)
    
    if super_cat and super_cat in multi_config.get("super_category_configs", {}):
        return multi_config["super_category_configs"][super_cat]
    
    raise ValueError(f"No config found for class '{class_name}' (super category: '{super_cat}') in multi-config")


def create_multi_config_template(output_path: str, base_config: Dict[str, Any]) -> None:
    """
    Create a multi-config template file with all super categories.
    
    Args:
        output_path: Path to save the template
        base_config: Base config to use as template for each super category
    """
    multi_config = {
        "super_category_configs": {},
        "default": base_config,
    }
    
    for super_cat in COCO_SUPER_CATEGORY_LIST:
        multi_config["super_category_configs"][super_cat] = base_config.copy()
    
    with open(output_path, "w") as f:
        json.dump(multi_config, f, indent=2)
    
    print(f"Multi-config template saved to: {output_path}")


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a config and return list of warnings/issues.
    
    Args:
        config: Config dict to validate
        
    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []
    
    if "parameters" not in config:
        warnings.append("Missing 'parameters' key")
    elif "method" not in config["parameters"]:
        warnings.append("Parameters missing 'method' key")
    
    if "transition_matrix" not in config:
        warnings.append("Missing 'transition_matrix' key")
    elif "classes" in config:
        tm_classes = set(config["transition_matrix"].keys())
        config_classes = set(config["classes"])
        if tm_classes != config_classes:
            warnings.append(
                f"Transition matrix classes {tm_classes} don't match config classes {config_classes}"
            )
    
    if "classes" not in config:
        warnings.append("Missing 'classes' key")
    
    return warnings
