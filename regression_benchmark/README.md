# Regression Benchmark

End-to-end object detection benchmark: train and evaluate models on **original** vs. **corrected** (post-processed) labels, then compare performance.

---

## Directory Structure

```
regression_benchmark/
├── inference/       # Run pre-trained detectors on validation sets
├── training/        # Fine-tune models on original or corrected labels
├── calibration/     # Post-hoc confidence calibration (isotonic regression)
├── evaluation/      # Compute metrics (mAP, soft-label divergence)
└── utils/           # Shared utilities (plot layout)
```

---

## Supported Models

| Model | Type | Used For |
|-------|------|----------|
| **Faster R-CNN** | Two-stage detector | Training + Inference |
| **YOLOv8-L** | One-stage detector | Training + Inference |
| **RT-DETR** | Transformer detector | Training + Inference |
| **YOLO-World** | Open-vocab detector | Inference only |
| **Grounding DINO** | Open-vocab detector | Inference only |
| **OWL-ViT** | Open-vocab detector | Inference only |

---

## Workflow

Each module comes with an executable .sh script, for evaluation there are two .sh scripts.
In particular training requires substantial compute time ~ 2 days on a single GPU, see details below.

### 1. Training

Fine-tune COCO-pretrained models on other datasets. Compare fine-tuning results for hard and soft labels.
Note that running fine_tune.sh will take ~2 days on a single Nvidia A100, consider commenting out non-needed models/datasets for speedup or adapt the .sh script for multi GPU usage.

| Script | Model | Labels |
|--------|-------|--------|
| `fine_tune_frcnn_cityscapes.py` | Faster R-CNN | Original |
| `fine_tune_frcnn_cityscapes_ours_hard.py` | Faster R-CNN | Corrected |
| `fine_tune_yolov8l_cityscapes.py` | YOLOv8-L | Original |
| `fine_tune_yolov8l_cityscapes_ours_hard.py` | YOLOv8-L | Corrected |
| `fine_tune_rtdetr_cityscapes.py` | RT-DETR | Original |
| `fine_tune_rtdetr_cityscapes_ours_hard.py` | RT-DETR | Corrected |

Same pattern exists for `_kitti`, `_pascal`, and `_coco` variants.

Notebooks:
- `prep_cityscapes_data.ipynb` — Prepare Cityscapes for YOLO format
- `prep_kitti_data.ipynb` — Prepare KITTI (split + format conversion)
- `soft_labels_to_yolo.ipynb` — Convert soft-label datasets to YOLO format

### 2. Inference

Some models require training before inference!
Run pre-trained (or fine-tuned) detectors on validation sets to produce predictions in COCO format.

```bash
# Run all models on all datasets
bash regression_benchmark/inference/inference.sh
```

Key scripts:
- `inference.sh` / `additional_inference.sh` — Shell scripts orchestrating model inference
- `create_coco_subset.py` — Create COCO training subset for efficiency
- `convert_cityscapes_filenames_to_id.py` — Map Cityscapes filenames to numeric IDs

### 3. Calibration

Apply post-hoc calibration to model confidence scores using isotonic regression with k-fold cross-validation.

```bash
bash regression_benchmark/calibration/calibrate.sh
```

- `calibrate.sh` / `additional_calibrate.sh` — Run calibration for all models × datasets
- `gather_results.ipynb` — Aggregate calibration results

### 4. Evaluation

Compare model predictions against ground truth (hard-label mAP) and soft-label distributions (divergence metrics).

```bash
bash regression_benchmark/evaluation/evaluate_hard.sh
```

- `evaluate_hard.sh` — Compute standard mAP metrics
- `gather_results.ipynb` — Aggregate evaluation results across all model×dataset×label combinations

---

## Dataset Paths

Scripts reference dataset directories that need to be configured. Update paths at the top of each script:

```python
# Training scripts (YOLO/RT-DETR)
settings.update({"datasets_dir": "/path/to/datasets"})

# Faster R-CNN training
train_img_dir = "/path/to/datasets/Cityscapes/train/images"
train_ann_dir = "/path/to/datasets/Cityscapes/train/json"

# Shell scripts
--gt_file /path/to/datasets/Cityscapes/val/instances_val.json
--image_dir /path/to/datasets/Cityscapes/val/images
```

### Expected Dataset Layout

```
/path/to/datasets/
├── Cityscapes/
│   ├── train/  (images/, labels/, json/)
│   └── val/    (images/, labels/, instances_val.json)
├── KITTI/
│   ├── train/  (images/, labels/)
│   └── val/    (images/, labels/, instances_val.json)
├── VOC/
│   ├── images/ (train2012/, val2012/)
│   └── labels/ (train2012/, val2012/, instances_val2012.json)
└── COCO/2017/
    ├── train2017/, val2017/
    └── annotations/ (instances_val2017.json, instances_train2017.json)
```

---
