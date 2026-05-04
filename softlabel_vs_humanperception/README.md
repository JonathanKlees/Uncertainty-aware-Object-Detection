# Soft Labels vs Human Perception

A human annotation study comparing soft-label class probabilities against independent human agreement ratings.

---

## Study Design

Three reviewers independently rate how well a bounding-box class label matches the depicted object on a 0–1 scale:
- **0.0** = Completely wrong class
- **0.5** = Ambiguous / uncertain
- **1.0** = Clearly correct class

Each reviewer uses an annotation tool that displays the cropped bounding box region with the proposed class label, then records a keypress rating.

---

## Workflow

### 1. Sample Selection

```
random_sampling.ipynb
```
Randomly samples bounding boxes from each dataset's soft-label annotations for the human study.

**Output:** `samples/{Dataset}.json` — Sampled annotation objects per dataset

### 2. Annotation

Each reviewer runs their tool independently:

```bash
python tool_reviewer_1.py   # Configure DATASET_NAME and IMAGE_DIR at top
python tool_reviewer_2.py
python tool_reviewer_3.py
```

**Configuration (per tool):**
- `DATASET_NAME`: One of `Cityscapes`, `COCO`, `Kitti`, `PascalVOC`
- `IMAGE_DIR`: Path to raw dataset images
- `INPUT_JSON`: Path to sampled annotations (`samples/{Dataset}.json`)

**Controls:**
- Keys `^`, `1`–`9`, `0` map to scores 0.0–1.0
- `b` = go back to previous annotation
- `q` = quit (progress is saved)

**Output:** `review_decisions/{Dataset}_class_or_not_review_reviewer_{1,2,3}.json`

### 3. Merge Results

```bash
python merge_results.py
```

Or use `merge_results.ipynb` for interactive merging with additional analysis.

Combines all three reviewers' scores into a single merged file per dataset.

**Output:** `review_decisions/{Dataset}_class_or_not_review_merged.json`

---

## File Structure

```
softlabel_vs_humanperception/
├── tool_reviewer_1.py                # Annotation tool (reviewer 1)
├── tool_reviewer_2.py                # Annotation tool (reviewer 2)
├── tool_reviewer_3.py                # Annotation tool (reviewer 3)
├── merge_results.py                  # Merge reviewer decisions
├── merge_results.ipynb               # Interactive merging + analysis
├── random_sampling.ipynb             # Sample selection
├── samples/                          # Sampled annotations per dataset
│   ├── Cityscapes.json
│   ├── COCO.json
│   ├── Kitti.json
│   └── PascalVOC.json
└── review_decisions/                 # Reviewer outputs + merged
    ├── {Dataset}_class_or_not_review_reviewer_1.json
    ├── {Dataset}_class_or_not_review_reviewer_2.json
    ├── {Dataset}_class_or_not_review_reviewer_3.json
    └── {Dataset}_class_or_not_review_merged.json
```

---

## Output JSON Format

Each review decision file contains:

```json
{
  "all_samples": [
    {
      "box_id": "unique_id",
      "image_file": "filename.png",
      "bbox": [x1, y1, x2, y2],
      "proposed_class": "car",
      "human_class_agreement": {
        "reviewer_1": 0.9,
        "reviewer_2": 0.8,
        "reviewer_3": 1.0
      }
    }
  ]
}
```

The merged files contain all three reviewers' scores in `human_class_agreement`. Individual files contain only that reviewer's score.
