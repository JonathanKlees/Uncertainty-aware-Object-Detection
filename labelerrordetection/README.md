# Label Error Detection

Evaluates object detection model predictions against the label error ground truth (LEGT) and the validated ground truth (VGT).

---

## Directory Structure

```
labelerrordetection/
├── cascadercnn_preds/       # Cascade R-CNN prediction evaluation
│   └── eval_preds_vs_gt.py  # Evaluate predictions vs original GT (COCO metrics)
├── gd_preds/                # Grounding DINO prediction evaluation
│   ├── eval_preds_vs_gt.py              # Evaluate predictions vs GT
│   ├── plot_histogram_gd_area_vs_vgt_area.py   # Compare prediction vs VGT area distributions
│   └── plot_histogram_gd_pred_scores.py        # Plot prediction score distributions
└── plots/
    └── plot_histogram_preds_vs_vgt_area.py  # Compare both models' area distributions vs VGT
```

---

## Purpose

1. **Evaluate detection quality**: Run standard COCO evaluation (mAP, AP@IoU thresholds) for Cascade R-CNN and Grounding DINO predictions against original GT
2. **Area distribution analysis**: Compare bounding box area distributions between model predictions and VGT to understand detection coverage across object sizes
3. **Score distribution analysis**: Analyze Grounding DINO confidence scores across datasets and parameter settings

---

## Configuration

Each script defines paths to:
- **GT**: Original dataset ground truth (`/path/to/benchmark_datasets_gt/{dataset}/val_coco_format.json`)
- **VGT**: Validated ground truth (`/path/to/benchmark_datasets_vgt/{dataset}/val_coco_format.json`)
- **Predictions**: Model outputs (`/path/to/benchmark_datasets_predictions_second_round/{Model}/{dataset}/val_coco_format.json`)

Grounding DINO predictions are organized by `boxthresh_X_textthresh_Y/nms_Z/` subdirectories reflecting the detection parameters used.
