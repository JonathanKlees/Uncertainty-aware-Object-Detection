# Label Errors Module

This module creates, validates, counts, analyzes, and visualizes the **Label Error Ground Truth (LEGT)** — a dataset of identified annotation errors in object detection benchmarks.

The LEGT is created by comparing the original dataset ground truth (GT) against a crowd-validated ground truth (VGT) derived from soft-label annotations.

---

## Error Types

| Type | Code | Description |
|------|------|-------------|
| Missing | `missing` | Object present in VGT but absent from original GT |
| Misaligned | `misaligned` | Object in both GT and VGT, but bounding box IoU < threshold |
| Classification | `classification` | Object matched (IoU ≥ threshold) but class label differs |
| Original-only | `original_only` | Object in original GT but not in VGT (spurious annotation) |

---

## Pipeline Execution Order

### Step 1: Prepare Validated Ground Truth (VGT)

| Script | Purpose |
|--------|---------|
| `1_filter_hard_labels.py` | Convert soft labels → hard labels (prob > 0.5, exclude `cantsolve`). Saves per-image JSONs in COCO format. |
| `1_1_save_all_annotated_bboxes.py` | Save **all** annotated bounding boxes (no probability filter) for analysis of "original-only" errors. |

**Input:** `our_datasets/soft_*.json` (post-processed soft labels)
**Output:** Per-split directories with COCO-format JSONs (`val_coco_format.json`, per-image JSONs in `json/`)

### Step 2: Create Label Error Ground Truth

| Script | Purpose |
|--------|---------|
| `2_create_label_error_ground_truth_cityscapes.py` | Create LEGT for Cityscapes |
| `2_create_label_error_ground_truth_coco.py` | Create LEGT for COCO |
| `2_create_label_error_ground_truth_kitti.py` | Create LEGT for KITTI (includes "Don't Care" region handling via IoA) |
| `2_create_label_error_ground_truth_pascalvoc.py` | Create LEGT for PascalVOC |
| `2_1_count_vis_label_error_ground_truth_*.py` | Count errors and optionally visualize them per dataset |
| `2_2_delete_label_errors_with_wrong_bbox_format.py` | Remove entries with invalid bounding boxes (out-of-bounds, degenerate) |
| `2_3_create_label_error_ground_truth_variants.py` | Create filtered LEGT variants: Variant 1 (prob ≥ 0.5), Variant 2 (prob ≥ 0.8 AND height ≥ 40px) |

**Input:** GT directory, VGT directory, ALL_VGT directory
**Output:** Per-image LEGT JSON files categorizing each annotation by error type

### Step 3: Count & Analyze Errors

| Script | Purpose |
|--------|---------|
| `3_count_label_errors_different_configurations_cityscapes.py` | Count errors under different probability/height thresholds → CSV + LaTeX table |
| `3_count_label_errors_different_configurations_coco.py` | Same for COCO |
| `3_count_label_errors_different_configurations_kitti.py` | Same for KITTI (also filters by "Don't Care" overlap) |
| `3_count_label_errors_different_configurations_pascalvoc.py` | Same for PascalVOC |
| `3_1_original_only_estimation.py` | Compute confidence intervals for spurious/missing error rates using binomial proportions |
| `3_2_original_only_distribution_plot.py` | Plot log-scale area distribution of "original-only" boxes across all datasets |

**Output:** CSV files, LaTeX-formatted tables, distribution plots

### Step 4: Additional Analysis

| Script | Purpose |
|--------|---------|
| `4_difficult_labels_vs_label_errors_analysis_pascalvoc.py` | Analyze correlation between PascalVOC's "difficult" flag and original-only errors |

---

## Visualization & Paper Figures

| Script | Purpose |
|--------|---------|
| `plot_label_errors_paper.py` | Generate cropped paper figures of label errors (missing, misaligned, classification) with GT overlay |
| `plot_original_only_boxes_paper.py` | Generate cropped paper figures of original-only errors with VGT overlay |
| `plot_ambiguous_cases.py` | Visualize ambiguous tie cases with bounding boxes and class labels |

---

## Helper Modules

### `helpers_legt.py`

Core utility library for LEGT creation:
- `area(bbox)` — Compute bounding box area
- `intersection(box1, box2)` — Compute intersection area
- `compute_iou(box1, box2)` — Intersection over Union
- `compute_ioa(box1, box2)` — Intersection over Area (for "Don't Care" regions)
- Greedy box matching between GT and VGT
- Label error extraction and classification

### `helpers_vis.py`

Visualization helper:
- `vis_main_errors()` — Create 3-panel images (GT / errors / validated) for detected label errors
- Image cropping with configurable context factor
- Supports rendering on Cityscapes, KITTI, PascalVOC, COCO images

---

## Auxiliary Scripts

| Script | Purpose |
|--------|---------|
| `filter_ambiguous_cases.py` | Find tie cases where multiple classes have equal max probability in soft labels |
| `number_of_annotations.py` | Count how many annotations each bounding box received (11, 22, 33, or 44 reviews) |

---

## Configuration

Each script defines path variables at the top that need to be set to your local directories:

```python
GT = "/path/to/benchmark_datasets_gt/{DATASET}"          # Original dataset GT (COCO format)
VGT = "/path/to/benchmark_datasets_vgt/{DATASET}"        # Validated ground truth
ALL_VGT = "/path/to/benchmark_datasets_vgt_all_bboxes/{DATASET}"  # All annotated boxes
LEGT = "/path/to/benchmark_datasets_legt_second_round/{DATASET}"  # Output: label error GT
```

Key parameters:
- **IoU threshold** for matching: typically 0.5
- **Probability threshold** for hard labels: 0.5 (Variant 1) or 0.8 (Variant 2)
- **Height threshold**: 40px minimum (Variant 2 only)

---

## Output Format

Each LEGT JSON file (per image) contains a list of annotations with:

```json
{
  "annotations": [
    {
      "bbox": [x1, y1, x2, y2],
      "category": "car",
      "error_type": "missing",
      "probability": 0.85
    }
  ]
}
```
