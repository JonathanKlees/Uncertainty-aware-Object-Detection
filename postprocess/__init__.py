"""
Post-processing module for correcting biased annotation distributions.

Main components:
- data_loader: Load datasets from JSON files
- methods: Post-processing methods (CleverLabel, downweight, etc.)
- methods_factory: Method factory functions and parameter grids
- methods_wrappers: Method wrappers with smoothing
- evaluation: Evaluation utilities
- visualization: Plotting and printing utilities
- clever_labeling: Core CleverLabel algorithm
- smoothing: Smoothing utilities and type aliases
- metrics: Core divergence metrics (KL, JS)
- metrics_legacy: Legacy metrics that combine smoothing + metric
- metrics_comparison: Dataset comparison utilities
"""

from postprocess.smoothing import Counts, Dataset

__all__ = [
    "Counts",
    "Dataset",
]
