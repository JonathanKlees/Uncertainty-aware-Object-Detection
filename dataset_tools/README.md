# Dataset Tools

Jupyter notebooks for converting between annotation formats.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `convert_to_hard_labels.ipynb` | Convert soft-label annotation JSONs to hard labels using a probability threshold. Uses COCO annotation format as reference for image metadata. |
| `convert_yolo_to_coco.ipynb` | Convert YOLO-format label files (`.txt` per image) to COCO JSON format. Covers Cityscapes, PascalVOC, and KITTI datasets. |
| `transfer_image_info.ipynb` | Transfer image metadata (dimensions, file names, IDs) from one COCO annotation file to another. Useful when merging or aligning annotation sources. |

---

## Usage

Each notebook has dataset-specific paths that need to be configured in the first cells. Update the `image_dir`, `label_dir`, and output paths to match your local dataset layout.
