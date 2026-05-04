"""
Core post-processing methods for correcting biased annotation distributions.

All methods work on probability distributions (probs -> probs).
For legacy counts-based methods, see methods_legacy.py.
"""

import random
from typing import Callable, Dict, List, Optional

from postprocess.clever_labeling import cleverlabel
from postprocess.smoothing import Distribution, DistDataset


# Type alias for probs-based post-processing methods
ProbsMethod = Callable[[DistDataset], DistDataset]


# =============================================================================
# Probs-based methods (probs -> probs)
# =============================================================================

def identity(map_id_probs: DistDataset) -> DistDataset:
    """Identity method - returns input unchanged."""
    return map_id_probs


def perfect(
    map_id_probs: DistDataset,
    map_id_unbiased_probs: DistDataset,
) -> DistDataset:
    """Perfect method - returns ground truth probs."""
    return map_id_unbiased_probs


def resample(
    map_id_probs: DistDataset,
    classes: List[str],
    n_samples: int = 11, # common value for annotator counts
    seed: int = 0,
) -> DistDataset:
    """
    Resample from probability distribution using multinomial sampling.
    
    Takes n_samples from the distribution and converts back to probs.
    
    Args:
        map_id_probs: Object ID -> probability distribution mapping
        classes: List of class names (for ordering)
        n_samples: Number of samples to draw
        seed: Random seed
    """
    rng = random.Random(seed)
    out: DistDataset = {}
    
    for k, probs in map_id_probs.items():
        # Build probability list aligned with classes
        prob_list = [probs.get(c, 0.0) for c in classes]
        total = sum(prob_list)
        if total <= 0:
            out[k] = {c: 0.0 for c in classes}
            continue
        
        # Normalize
        prob_list = [p / total for p in prob_list]
        
        # Multinomial sampling using random.choices
        samples = rng.choices(classes, weights=prob_list, k=n_samples)
        
        # Convert samples back to distribution
        counts = {c: 0 for c in classes}
        for s in samples:
            counts[s] += 1
        
        new_probs = {c: counts[c] / n_samples for c in classes}
        out[k] = new_probs
    
    return out


def downweight_proposed_class(
    map_id_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    downweight_factor: float = 0.9,
) -> DistDataset:
    """
    Downweight the probability of the proposed class and renormalize.
    
    Args:
        map_id_probs: Object ID -> probability distribution mapping
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of all class names
        downweight_factor: Factor to multiply proposed class prob by (0-1)
    """
    out: DistDataset = {}
    
    for k, probs in map_id_probs.items():
        proposed_cls = map_id_proposed_class.get(k)

        # print(proposed_cls, probs)
        if proposed_cls is None:
            out[k] = probs
            continue
        
        # Downweight proposed class
        new_probs = {}
        for cls in classes:
            p = probs.get(cls, 0.0)
            if cls == proposed_cls:
                new_probs[cls] = p * downweight_factor
                # print(f"Downweighting proposed class '{cls}' from {p:.4f} to {new_probs[cls]:.4f}")
            else:
                new_probs[cls] = p
        
        # Renormalize, if no other probability mass exists then this removes previous effect
        # even with some mass, the class es rebalanced 
        total = sum(new_probs.values())
        if total > 0:
            new_probs = {c: p / total for c, p in new_probs.items()}
        
        out[k] = new_probs

        # print(new_probs)
    
    return out


def apply_per_class_damper(
    map_id_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    per_class_dampers: Dict[str, float],
    renorm_mode: str = "full",
) -> DistDataset:
    """
    Apply per-class damper weights to correct biased probabilities.
    
    Formula: new_prob[proposed] = damper[proposed_cls] * biased_prob[proposed]
    Such that: biased * damper ≈ unbiased
    
    Args:
        map_id_probs: Object ID -> probability distribution mapping
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of all class names
        per_class_dampers: Dict mapping class name -> damper weight (from training)
        renorm_mode: Renormalization mode:
            - "full": Renormalize ALL probabilities (like downweight)
            - "remaining": Keep damper-adjusted proposed prob fixed, 
                           rescale only remaining classes
    
    Returns:
        DistDataset with corrected probability distributions
    """
    out: DistDataset = {}
    
    for k, probs in map_id_probs.items():
        proposed_cls = map_id_proposed_class.get(k)
        
        if proposed_cls is None:
            out[k] = probs
            continue
        
        # Get damper for this proposed class (default to 1.0 if unseen)
        damper = per_class_dampers.get(proposed_cls, 1.0)
        
        # Get current proposed class probability
        proposed_prob = probs.get(proposed_cls, 0.0)
        
        # Apply damper to proposed class
        new_proposed_prob = proposed_prob * damper
        
        # Clip to valid probability range
        new_proposed_prob = max(0.0, min(1.0, new_proposed_prob))
        
        if renorm_mode == "full":
            # Variant A: Full renormalization (like downweight)
            # Apply damper then renormalize all probabilities
            new_probs = {}
            for cls in classes:
                if cls == proposed_cls:
                    new_probs[cls] = new_proposed_prob
                else:
                    new_probs[cls] = probs.get(cls, 0.0)
            
            total = sum(new_probs.values())
            if total > 0:
                new_probs = {c: p / total for c, p in new_probs.items()}
        
        elif renorm_mode == "remaining":
            # Variant B: Remaining renormalization (like cleverlabel)
            # Keep damper-adjusted proposed prob fixed, rescale remaining classes
            remaining_mass = 1.0 - new_proposed_prob
            
            # Sum of old remaining (non-proposed) probabilities
            old_remaining = sum(probs.get(c, 0.0) for c in classes if c != proposed_cls)
            
            new_probs = {}
            new_probs[proposed_cls] = new_proposed_prob
            
            if old_remaining > 0 and remaining_mass > 0:
                scale = remaining_mass / old_remaining
                for cls in classes:
                    if cls != proposed_cls:
                        new_probs[cls] = probs.get(cls, 0.0) * scale
            elif remaining_mass > 0:
                # All other classes were 0, distribute uniformly
                n_other = len(classes) - 1
                for cls in classes:
                    if cls != proposed_cls:
                        new_probs[cls] = remaining_mass / n_other if n_other > 0 else 0.0
            else:
                # No remaining mass (proposed takes all)
                for cls in classes:
                    if cls != proposed_cls:
                        new_probs[cls] = 0.0
        
        else:
            raise ValueError(f"Unknown renorm_mode: {renorm_mode}")
        
        out[k] = new_probs
    
    return out


def apply_cleverlabel(
    map_id_probs: DistDataset,
    map_id_proposed_class: Dict[str, str],
    classes: List[str],
    transition_c: Dict[str, List[float]],
    delta: float = 0.1,
    mu: float = 0.75,
    one_star: float = 0.99,
    avoid_overcorrection_threshold: float = 0,
) -> DistDataset:
    """
    Apply CleverLabel bias correction on probability distributions.
    
    Args:
        map_id_probs: Object ID -> probability distribution mapping
        map_id_proposed_class: Object ID -> proposed class mapping
        classes: List of class names
        transition_c: Transition matrix for class blending
        delta: Bias correction parameter
        mu: Class blending weight (1.0 = no blending)
        one_star: One-star acceptance probability
        avoid_overcorrection_threshold: Threshold to avoid overcorrection
    
    Returns:
        DistDataset with corrected probability distributions
    """
    out: DistDataset = {}
    
    for k, probs in map_id_probs.items():
        proposed_cls = map_id_proposed_class.get(k)
        if proposed_cls is None:
            out[k] = probs
            continue

        result = cleverlabel(
            probs, 
            proposal_class=proposed_cls, 
            classes=classes, 
            delta=delta, 
            one_star=one_star, 
            mu=mu, 
            avoid_overcorrection_threshold=avoid_overcorrection_threshold,
            transition_c=transition_c,
            apply_bc=True if mu > 0 else False, 
            apply_cb=True if mu < 1 else False,
            return_counts=False,
            input_probs=True,  # Probs input
        )
        out[k] = result
    
    return out
