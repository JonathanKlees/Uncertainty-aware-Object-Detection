# Our Datasets

Soft-label annotation datasets used throughout the benchmark. These contain crowd-sourced bounding box annotations with class probability distributions.

Download the datasets from the Harvard Dataverse (https://doi.org/10.7910/DVN/L7QCUG) and paste them into this folder. Run the Juypter Notebook to split the datasets into train / val.

---

## File Naming Convention

```
soft_{Dataset}_{split}.json
```

Examples:
- `soft_Cityscapes_train.json` — Cityscapes training split
- `soft_COCO_2017_val.json` — COCO 2017 validation split
- `soft_PascalVOC_2012_detection_trainval.json` — PascalVOC 2012 full trainval

---

## JSON Schema

Each file contains a list of annotated bounding boxes with soft-label distributions:

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 123,
      "bbox": [x, y, width, height],
      "category_id": 1,
      "soft_label": {
        "car": 0.7,
        "van": 0.2,
        "truck": 0.1
      },
      "num_annotations": 22
    }
  ],
  "categories": [...]
}
```

Key fields:
- `soft_label`: Probability distribution over classes (sums to 1.0)
- `num_annotations`: How many annotators reviewed this bounding box
- `bbox`: Bounding box in COCO format `[x, y, width, height]`

---

## Subdirectories

| Directory | Content |
|-----------|---------|
| `splits/` | Train/val split assignments (e.g., `PascalVOC_train_val_split.json`, `Kitti_train_val_split.json`) |
| `unbiased_samples/` | Subsets with unbiased (multi-annotator) samples used for post-processing tuning |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `split.ipynb` | Create train/val splits for all datasets locally |
