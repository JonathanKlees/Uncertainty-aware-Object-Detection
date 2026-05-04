import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def bbox_area_xywh(bbox):

    if bbox is None or len(bbox) < 4:
        return None

    w = float(bbox[2])
    h = float(bbox[3])

    if w <= 0 or h <= 0:
        return None

    return w * h


def extract_coco_prediction_areas(pred_file):

    preds = load_json(pred_file)

    areas = []
    for pred in preds:
        area = bbox_area_xywh(pred.get("bbox"))
        if area is not None:
            areas.append(area)

    return np.array(areas)


def extract_legt_areas(legt_file):
    data = load_json(legt_file)

    if isinstance(data, dict):
        if "annotations" in data:
            entries = data["annotations"]
        elif "errors" in data:
            entries = data["errors"]
        elif "objects" in data:
            entries = data["objects"]
        else:
            raise ValueError(
                f"Unknown LEGT format in {legt_file}. "
                f"Expected key 'annotations', 'errors', or 'objects'."
            )
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unknown JSON structure in {legt_file}")

    areas = []
    for entry in entries:
        area = bbox_area_xywh(entry.get("bbox"))
        if area is not None:
            areas.append(area)

    return np.array(areas)


def plot_area_histogram(dataset_name, pred_areas, vgt_areas, out_dir, bins=50):
    os.makedirs(out_dir, exist_ok=True)

    # log scale ist bei bbox areas meistens sinnvoll
    all_areas = np.concatenate([pred_areas, vgt_areas])
    all_areas = all_areas[all_areas > 0]

    min_area = max(1.0, np.min(all_areas))
    max_area = np.max(all_areas)

    log_bins = np.logspace(np.log10(min_area), np.log10(max_area), bins)

    plt.figure(figsize=(7, 5))

    plt.hist(
        pred_areas,
        bins=log_bins,
        alpha=0.5,
        color="#1b5e20",
        label=f"Predictions Grounding DINO",
    )

    plt.hist(
        vgt_areas,
        bins=log_bins,
        alpha=0.5,
        color="#ec6cbb",
        label=f"Validated Ground Truth",
    )

    plt.xscale("log")
    plt.xlabel("Bounding box area in pixels")
    plt.ylabel("Frequency")
    plt.grid()
    #plt.title(f"Bounding box area distribution: {dataset_name}")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{dataset_name}_bbox_area_gd_preds_vs_vgt.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_all_datasets_combined(datasets, out_dir, bins=50):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for dataset_name, paths in datasets.items():
        pred_areas = extract_coco_prediction_areas(paths["predictions"])

        pred_areas = pred_areas[pred_areas > 0]

        plt.hist(
            pred_areas,
            bins=bins,
            alpha=0.35,
            label=f"{dataset_name} ({len(pred_areas):,})",
        )

    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency")
    #plt.title("Grounding DINO bounding box area distributions")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, "all_datasets_gdino_bbox_area_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    datasets = {
        "PascalVOC": {
            "predictions": "/path/to/benchmark_datasets_predictions/GroundingDino/PascalVOC/boxthresh_0.2_textthresh_0.2/nms_0.7/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/PascalVOC/val_coco_format.json",
        },
        "COCO": {
            "predictions": "/path/to/benchmark_datasets_predictions/GroundingDino/COCO/boxthresh_0.25_textthresh_0.25/nms_0.6/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/COCO/val_coco_format.json",
        },
        "Cityscapes": {
            "predictions": "/path/to/benchmark_datasets_predictions/GroundingDino/Cityscapes/boxthresh_0.2_textthresh_0.2/nms_0.5/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/Cityscapes/val_coco_format.json",
        },
        "KITTI": {
            "predictions": "/path/to/benchmark_datasets_predictions/GroundingDino/Kitti/boxthresh_0.2_textthresh_0.2/nms_0.5/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/Kitti/val_coco_format.json",
        },
    }

    out_dir = "bbox_area_histograms"

    for dataset_name, paths in datasets.items():
        pred_areas = extract_coco_prediction_areas(paths["predictions"])
        vgt_areas = extract_legt_areas(paths["vgt"])

        print(dataset_name)
        print(f"  GD predictions: {len(pred_areas):,}")
        print(f"  VGT boxes:     {len(vgt_areas):,}")
        print(f"  GD mean area:   {np.mean(pred_areas):.2f}")
        print(f"  VGT mean area: {np.mean(vgt_areas):.2f}")

        plot_area_histogram(
            dataset_name=dataset_name,
            pred_areas=pred_areas,
            vgt_areas=vgt_areas,
            out_dir=out_dir,
            bins=60,
        )

    plot_all_datasets_combined(datasets, out_dir=out_dir, bins=60)


if __name__ == "__main__":
    main()