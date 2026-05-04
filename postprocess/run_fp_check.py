#!/usr/bin/env python3
"""
False Positive (FP) Check for human calibration data.

Compares averaged review results against biased/unbiased/postprocessed distributions
and individual reviewer results to compute acceptance/rejection statistics.

Logic:
- avg_review >= threshold: acceptance for drawn_class
- avg_review < threshold: refusal for drawn_class
- If drawn_class matches majority class of distribution: count TP (accepted) or FP (refused)
- If drawn_class != majority class: count as NA (not applicable/discarded)

For individual reviewers:
- Each reviewer score is compared separately using the same threshold
- If reviewer accepts (score >= threshold) and avg accepts: agreement
- If reviewer refuses (score < threshold) and avg refuses: agreement

Usage:
    # Single dataset
    python -m postprocess.run_fp_check --dataset pascalvoc --threshold 0.5

    # With custom alpha for Dirichlet smoothing
    python -m postprocess.run_fp_check --dataset pascalvoc --threshold 0.5 --alpha 0.5

    # COCO (merges all super-category files)
    python -m postprocess.run_fp_check --dataset coco --threshold 0.5
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from postprocess.calibration_utils import (
    ObjectCalibrationData,
    classify_review_bucket,
    load_dataset,
    load_default_config,
    extract_object_calibration_data,
    get_default_data_dir,
    get_default_aggregated_dir,
    load_aggregated_index,
)


@dataclass
class FPStatistics:
    """Statistics for a single comparison type."""
    name: str  # e.g., "biased", "unbiased", "postprocessed", "individual_reviewer"
    tp: int = 0  # True positive: classes match AND accepted
    fp: int = 0  # False positive: classes match AND refused
    na: int = 0  # Not applicable: classes don't match
    
    @property
    def total_matching(self) -> int:
        """Total cases where classes matched."""
        return self.tp + self.fp
    
    @property
    def total(self) -> int:
        """Total cases evaluated."""
        return self.tp + self.fp + self.na
    
    @property
    def acceptance_rate(self) -> float:
        """Acceptance rate among matching cases."""
        if self.total_matching == 0:
            return 0.0
        return self.tp / self.total_matching
    
    @property
    def match_rate(self) -> float:
        """Rate of cases where drawn_class matched majority."""
        if self.total == 0:
            return 0.0
        return self.total_matching / self.total


@dataclass
class FPCheckResult:
    """Complete FP check results for a dataset."""
    dataset: str
    threshold: float
    n_objects: int
    
    # Statistics per comparison type
    biased: FPStatistics = field(default_factory=lambda: FPStatistics("biased"))
    unbiased: FPStatistics = field(default_factory=lambda: FPStatistics("unbiased"))
    dirichlet: FPStatistics = field(default_factory=lambda: FPStatistics("dirichlet"))
    postprocessed: FPStatistics = field(default_factory=lambda: FPStatistics("postprocessed"))
    postprocessed_selected: FPStatistics = field(default_factory=lambda: FPStatistics("postprocessed_selected"))
    individual_reviewers: FPStatistics = field(default_factory=lambda: FPStatistics("individual_reviewers"))


def compute_fp_statistics(
    object_data_list: List[ObjectCalibrationData],
    threshold: float,
) -> FPCheckResult:
    """
    Compute FP statistics by comparing averaged review scores against distributions.
    
    Args:
        object_data_list: List of ObjectCalibrationData with calibration info
        threshold: Acceptance threshold (avg_review >= threshold means acceptance)
    
    Returns:
        FPCheckResult with statistics for each comparison type
    """
    result = FPCheckResult(
        dataset="",
        threshold=threshold,
        n_objects=len(object_data_list),
    )
    
    for obj_data in object_data_list:
        drawn_class = obj_data.drawn_class
        avg_review = obj_data.avg_review_score
        
        # Determine if avg review accepts or refuses drawn_class
        avg_accepts = avg_review >= threshold
        
        # Compare against biased majority
        _update_stats(
            result.biased,
            drawn_class=drawn_class,
            majority_class=obj_data.biased_majority_class,
            accepted=avg_accepts,
        )
        
        # Compare against unbiased majority
        _update_stats(
            result.unbiased,
            drawn_class=drawn_class,
            majority_class=obj_data.unbiased_majority_class,
            accepted=avg_accepts,
        )        

        
        # Compare against postprocessed majority (if available)
        if obj_data.postprocessed_majority_class:
            _update_stats(
                result.postprocessed,
                drawn_class=drawn_class,
                majority_class=obj_data.postprocessed_majority_class,
                accepted=avg_accepts,
            )
        
        # Compare against postprocessed_selected majority (if available)
        if obj_data.postprocessed_selected_majority_class:
            _update_stats(
                result.postprocessed_selected,
                drawn_class=drawn_class,
                majority_class=obj_data.postprocessed_selected_majority_class,
                accepted=avg_accepts,
            )
        
        # Compare individual reviewers
        # Each reviewer's decision vs avg decision
        for reviewer_score in obj_data.review_results:
            reviewer_accepts = reviewer_score >= threshold

            _update_stats(
                result.individual_reviewers,
                 drawn_class=drawn_class,
                # For individual reviewers, we consider "match" if drawn_class matches majority (which is drawn_class if avg accepts)
                 majority_class=drawn_class if reviewer_accepts == avg_accepts else None,
                 accepted=avg_accepts,            
            )
            
            # # For individual reviewers, we compare reviewer decision to avg decision
            # # "Match" means reviewer agrees with the avg decision
            # if reviewer_accepts == avg_accepts:
            #     # Agreement - this is like "class match"
            #     if avg_accepts:
            #         result.individual_reviewers.tp += 1  # Both accept
            #     else:
            #         result.individual_reviewers.fp += 1  # Both refuse (still "FP" in this naming)
            # else:
            #     # Disagreement
            #     result.individual_reviewers.na += 1
    
    return result


def _update_stats(
    stats: FPStatistics,
    drawn_class: str,
    majority_class: str,
    accepted: bool,
) -> None:
    """Update statistics based on class match and acceptance."""
    if drawn_class == majority_class:
        if accepted:
            stats.tp += 1
        else:
            stats.fp += 1
    else:
        stats.na += 1


def print_statistics_table(result: FPCheckResult) -> None:
    """Print formatted statistics table."""
    print(f"\n{'='*70}")
    print(f"FP CHECK RESULTS: {result.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Threshold: {result.threshold}")
    print(f"Objects evaluated: {result.n_objects}")
    print()
    
    # Header
    print(f"{'Comparison Type':<20} {'TP':<8} {'FP':<8} {'NA':<8} {'Total':<8} {'Accept%':<10} {'Match%':<10}")
    print("-" * 70)
    
    # Rows
    for stats in [result.biased, result.unbiased, result.postprocessed, result.postprocessed_selected, result.individual_reviewers]:
        if stats.total > 0:
            print(f"{stats.name:<20} {stats.tp:<8} {stats.fp:<8} {stats.na:<8} "
                  f"{stats.total:<8} {stats.acceptance_rate*100:>6.1f}%   {stats.match_rate*100:>6.1f}%")
    
    print("-" * 70)
    
    # Individual reviewers (different interpretation)
    ir = result.individual_reviewers
    if ir.total > 0:
        print(f"\nIndividual Reviewers vs Consensus (avg):")
        print(f"  Agree Accept:  {ir.tp:>6}  ({ir.tp/ir.total*100:.1f}%)")
        print(f"  Agree Refuse:  {ir.fp:>6}  ({ir.fp/ir.total*100:.1f}%)")
        print(f"  Disagree:      {ir.na:>6}  ({ir.na/ir.total*100:.1f}%)")
        print(f"  Total votes:   {ir.total:>6}")
        agreement_rate = (ir.tp + ir.fp) / ir.total * 100 if ir.total > 0 else 0
        print(f"  Agreement rate: {agreement_rate:.1f}%")
    
    print(f"\n{'='*70}")


def print_detailed_explanation() -> None:
    """Print explanation of metrics."""
    print("""
Metric Explanation:
-------------------
- TP (Accepted): drawn_class == majority_class AND avg_review >= threshold
  (The human reviewers accepted the drawn class, and it matches the majority)

- FP (Refused): drawn_class == majority_class AND avg_review < threshold
  (The human reviewers refused the drawn class, but it matches the majority)

- NA (Not Matching): drawn_class != majority_class
  (The drawn class doesn't match the majority, so acceptance is not meaningful)

- Accept%: TP / (TP + FP) - Among matching cases, how often was it accepted
- Match%: (TP + FP) / Total - How often did drawn_class match majority

For Individual Reviewers:
- Compares each reviewer's decision to the averaged decision
- "Agree Accept/Refuse" indicates agreement with the consensus
- "Disagree" indicates the reviewer differed from the consensus
""")


# ── Bucket Confusion Matrix ──────────────────────────────────────────

BUCKET_NAMES = ["no", "maybe", "clearly"]
PROB_TYPES = ["bias", "unbias", "post.", "select"]


def _get_drawn_class_prob(
    obj_data: ObjectCalibrationData,
    prob_type: str,
) -> Optional[float]:
    """Get the drawn_class probability for a given distribution type."""
    if prob_type == "bias":
        return obj_data.biased_prob
    elif prob_type == "unbias":
        return obj_data.unbiased_prob
    elif prob_type == "post.":
        return obj_data.postprocessed_prob
    elif prob_type == "select":
        return obj_data.postprocessed_selected_prob
    return None


def compute_avg_vs_prob_bucket_matrices(
    object_data_list: List[ObjectCalibrationData],
    bucket_thresholds: Tuple[float, float] = (0.25, 0.75),
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[str, str, str]]]]:
    """
    Compute 3x3 confusion matrices: avg review bucket vs probability bucket.

    Also collects identifiers for error 2+ cases (off-by-2 bucket disagreements).

    Returns:
        (matrices dict, error2_ids dict)
        error2_ids maps prob_type -> list of (obj_id, review_bucket, prob_bucket)
    """
    bucket_idx = {name: i for i, name in enumerate(BUCKET_NAMES)}
    matrices: Dict[str, np.ndarray] = {}
    error2_ids: Dict[str, List[Tuple[str, str, str]]] = {}

    for prob_type in PROB_TYPES:
        matrix = np.zeros((3, 3), dtype=int)
        ids = []

        for obj_data in object_data_list:
            prob = _get_drawn_class_prob(obj_data, prob_type)
            if prob is None:
                continue

            review_bucket = classify_review_bucket(obj_data.avg_review_score, bucket_thresholds)
            prob_bucket = classify_review_bucket(prob, bucket_thresholds)

            r = bucket_idx[review_bucket]
            c = bucket_idx[prob_bucket]
            matrix[r, c] += 1

            # Off-by-2: corners (no-clearly or clearly-no)
            if abs(r - c) >= 2:
                ids.append((obj_data.obj_id, review_bucket, prob_bucket))

        if matrix.sum() > 0:
            matrices[prob_type] = matrix
            error2_ids[prob_type] = ids

    return matrices, error2_ids


def compute_reviewer_vs_avg_bucket_matrices(
    object_data_list: List[ObjectCalibrationData],
    bucket_thresholds: Tuple[float, float] = (0.25, 0.75),
) -> Dict[str, np.ndarray]:
    """
    Compute 3x3 confusion matrices: individual reviewer bucket vs avg review bucket.

    Produces one matrix per reviewer position (reviewer_1, reviewer_2, etc.)
    plus an "all_reviewers" aggregate matrix.

    Rows = individual reviewer bucket, Cols = avg review score bucket.
    All objects contribute (no data discarded).
    """
    bucket_idx = {name: i for i, name in enumerate(BUCKET_NAMES)}

    max_reviewers = max(len(obj.review_results) for obj in object_data_list)

    matrices: Dict[str, np.ndarray] = {}
    for r_idx in range(max_reviewers):
        matrices[f"reviewer_{r_idx + 1}"] = np.zeros((3, 3), dtype=int)
    matrices["all_reviewers"] = np.zeros((3, 3), dtype=int)

    for obj_data in object_data_list:
        avg_bucket = classify_review_bucket(obj_data.avg_review_score, bucket_thresholds)
        col = bucket_idx[avg_bucket]

        for r_idx, reviewer_score in enumerate(obj_data.review_results):
            rev_bucket = classify_review_bucket(reviewer_score, bucket_thresholds)
            row = bucket_idx[rev_bucket]

            matrices[f"reviewer_{r_idx + 1}"][row, col] += 1
            matrices["all_reviewers"][row, col] += 1

    return {k: v for k, v in matrices.items() if v.sum() > 0}


def compute_confusion_metrics(matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics from a 3x3 confusion matrix.
    
    Returns dict with:
        bucket_accuracy: fraction on diagonal (exact match)
        error_1plus: fraction off by 1 or more buckets (= 1 - bucket_accuracy)
        error_2plus: fraction off by 2 buckets (corners only)
        cohens_kappa: chance-corrected agreement
    """
    total = matrix.sum()
    if total == 0:
        return {"bucket_accuracy": 0.0, "error_1plus": 0.0, "error_2plus": 0.0, "cohens_kappa": 0.0}
    
    n = matrix.shape[0]
    
    # Diagonal = exact match
    diagonal = sum(matrix[i, i] for i in range(n))
    
    # Off-by-2 = corners (distance >= 2)
    off_by_2 = matrix[0, n-1] + matrix[n-1, 0]
    
    p_o = diagonal / total
    bucket_accuracy = p_o
    error_1plus = 1.0 - bucket_accuracy
    error_2plus = off_by_2 / total
    
    # Cohen's Kappa
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    p_e = sum(row_totals[i] * col_totals[i] for i in range(n)) / (total * total)
    
    if p_e >= 1.0:
        cohens_kappa = 1.0
    else:
        cohens_kappa = (p_o - p_e) / (1.0 - p_e)
    
    return {
        "bucket_accuracy": bucket_accuracy,
        "error_1plus": error_1plus,
        "error_2plus": error_2plus,
        "cohens_kappa": cohens_kappa,
    }


def print_confusion_metrics(matrix: np.ndarray) -> None:
    """Print metrics for a confusion matrix."""
    m = compute_confusion_metrics(matrix)
    print(f"  Bucket accuracy: {m['bucket_accuracy']*100:.1f}%  \n"
          f"  Error 1+: {m['error_1plus']*100:.1f}%  \n  "
          f"Error 2+: {m['error_2plus']*100:.1f}%  \n  "
          f"Cohen's Kappa: {m['cohens_kappa']:.3f}")


def print_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    row_label: str = "Reviewer",
    col_label: str = "Avg",
) -> None:
    """Print a formatted 3x3 confusion matrix with totals."""
    print(f"\n  {title}")
    print(f"  {'-'*50}")

    # Header
    header = f"  {row_label + ' vs ' + col_label:<20}"
    for name in BUCKET_NAMES:
        header += f" {name:>10}"
    header += f" {'| Total':>10}"
    print(header)
    print(f"  {'-'*50}")

    # Rows
    for i, row_name in enumerate(BUCKET_NAMES):
        row_str = f"  {row_name:<20}"
        for j in range(3):
            row_str += f" {matrix[i, j]:>10}"
        row_str += f" |{matrix[i, :].sum():>9}"
        print(row_str)

    # Column totals
    print(f"  {'-'*50}")
    totals_str = f"  {'Total':<20}"
    for j in range(3):
        totals_str += f" {matrix[:, j].sum():>10}"
    totals_str += f" |{matrix.sum():>9}"
    print(totals_str)


def print_all_confusion_matrices(
    prob_matrices: Dict[str, np.ndarray],
    reviewer_matrices: Dict[str, np.ndarray],
    error2_ids: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
) -> None:
    """Print all confusion matrices."""
    print(f"\n{'='*70}")
    print("BUCKET CONFUSION MATRICES (avg review vs drawn_class probability)")
    print(f"{'='*70}")
    print("Rows = avg review score bucket, Columns = probability bucket")
    print("Buckets: no (<=low), maybe (between), clearly (>=high)")

    for prob_type, matrix in prob_matrices.items():
        print_confusion_matrix(
            matrix,
            title=f"{prob_type.upper()}",
            row_label="Avg. Reviewer",
            col_label=prob_type,
        )
        print_confusion_metrics(matrix)
        # Print error 2+ identifiers
        if error2_ids and prob_type in error2_ids and error2_ids[prob_type]:
            ids = error2_ids[prob_type]
            print(f"  Error 2+ instances ({len(ids)}):")
            for obj_id, rev_bucket, prob_bucket in ids:
                print(f"    {obj_id}  (review={rev_bucket}, {prob_type}={prob_bucket})")

    print(f"\n{'='*70}")
    print("BUCKET CONFUSION MATRICES (individual reviewer vs avg review score)")
    print(f"{'='*70}")
    print("Rows = individual reviewer bucket, Columns = avg review score bucket")

    for label, matrix in reviewer_matrices.items():
        print_confusion_matrix(
            matrix,
            title=f"{label.upper().replace('_', ' ')}",
            row_label=label.split('_')[0].capitalize(),
            col_label="Avg.",
        )
        print_confusion_metrics(matrix)


def main():
    parser = argparse.ArgumentParser(
        description="Compute FP statistics for human calibration data"
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cityscapes", "kitti", "pascalvoc", "coco"],
        help="Dataset to load (coco merges all super-category files)"
    )
    parser.add_argument(
        "--data_dir",
        default=get_default_data_dir(),
        help="Directory containing unbiased_samples/"
    )
    parser.add_argument(
        "--source",
        choices=["unbiased", "aggregated"],
        default="aggregated",
        help="Data source for biased/postprocessed distributions. "
             "'unbiased': compute on-the-fly from unbiased_samples. "
             "'aggregated': load from final aggregated files (soft_*.json)."
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print detailed explanation of metrics"
    )
    parser.add_argument(
        "--bucket_thresholds",
        type=float,
        nargs=2,
        default=[0.25, 0.75],
        metavar=("LOW", "HIGH"),
        help="Bucket thresholds (default: 0.25 0.75). <=LOW=no, >=HIGH=clearly, between=maybe."
    )
    
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"FP CHECK: {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Source: {args.source}")
    print(f"Bucket thresholds: {args.bucket_thresholds[0]}, {args.bucket_thresholds[1]}")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    data, classes, is_coco, obj_to_super_category = load_dataset(args.data_dir, args.dataset)
    print(f"  Images: {len(data.get('objects', {}))}")
    print(f"  Classes: {len(classes)}")
    
    # Load aggregated index if requested
    aggregated_index = None
    if args.source == "aggregated":
        print(f"\nLoading aggregated index...")
        aggregated_index = load_aggregated_index(get_default_aggregated_dir(), args.dataset)
    
    # Load config
    print(f"\nLoading config...")
    config, multi_config = load_default_config(args.dataset)
    if config:
        print(f"  Method: {config.get('parameters', {}).get('method', 'unknown')}")
        if multi_config:
            print(f"  Multi-config with {len(multi_config.get('super_category_configs', {}))} super-categories")
    else:
        print("  Warning: No config loaded, postprocessed will be skipped")
    
    # Extract calibration data
    print("\nExtracting calibration data...")
    object_data_list = extract_object_calibration_data(
        data=data,
        classes=classes,
        alpha=0,
        is_coco=is_coco,
        obj_to_super_category=obj_to_super_category,
        config=config,
        multi_config=multi_config,
        aggregated_index=aggregated_index,
    )
    print(f"  Objects with calibration: {len(object_data_list)}")
    
    if not object_data_list:
        print("\nNo objects with calibration data found.")
        return
    
    bucket_thresh = tuple(args.bucket_thresholds)
    
    # Compute statistics
    print("\nComputing FP statistics...")
    result = compute_fp_statistics(object_data_list, bucket_thresh[1])
    result.dataset = args.dataset
    
    # Print results
    print_statistics_table(result)
    
    # Compute and print bucket confusion matrices
    
    prob_matrices, error2_ids = compute_avg_vs_prob_bucket_matrices(
        object_data_list, bucket_thresholds=bucket_thresh
    )
    reviewer_matrices = compute_reviewer_vs_avg_bucket_matrices(
        object_data_list, bucket_thresholds=bucket_thresh
    )
    print_all_confusion_matrices(prob_matrices, reviewer_matrices, error2_ids)
    
    if args.explain:
        print_detailed_explanation()
    
    print("DONE")


if __name__ == "__main__":
    main()
