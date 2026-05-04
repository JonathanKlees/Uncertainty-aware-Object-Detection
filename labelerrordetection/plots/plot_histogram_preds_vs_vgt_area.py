import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def bbox_area_xywh(bbox):
    """
    COCO bbox format: [x, y, width, height]
    """
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
        if pred["score"] >= 0.2:
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
            raise ValueError(f"Unknown LEGT format in {legt_file}")
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


def plot_area_histogram(dataset_name, gd_areas, cascade_areas, vgt_areas, out_dir, bins=50):
    os.makedirs(out_dir, exist_ok=True)

    all_areas = np.concatenate([gd_areas, cascade_areas, vgt_areas])
    all_areas = all_areas[all_areas > 0]

    min_area = max(1.0, np.min(all_areas))
    max_area = np.max(all_areas)

    log_bins = np.logspace(np.log10(min_area), np.log10(max_area), bins)

    plt.figure(figsize=(7, 5))

    plt.hist(
        gd_areas,
        bins=log_bins,
        alpha=0.5,
        color="#f57c00",  # orange
        #edgecolor="black",
        linewidth=0.5,
        label="Predictions Grounding DINO",
    )

    plt.hist(
        cascade_areas,
        bins=log_bins,
        alpha=0.5,
        color="#1565c0", 
        #edgecolor="black",
        linewidth=0.5,
        label="Predictions Cascade R-CNN",
    )

    plt.hist(
        vgt_areas,
        bins=log_bins,
        alpha=0.6,
        color="#d81b60",  
        #edgecolor="black",
        linewidth=0.5,
        label="Validated Ground Truth",
    )

    plt.xscale("log")
    plt.xlabel("Bounding box area in pixels")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{dataset_name}_bbox_area_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_all_datasets_combined(datasets, out_dir, bins=50):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for dataset_name, paths in datasets.items():
        gd_areas = extract_coco_prediction_areas(paths["gd_predictions"])
        cascade_areas = extract_coco_prediction_areas(paths["cascade_predictions"])

        plt.hist(
            gd_areas,
            bins=bins,
            alpha=0.3,
            label=f"{dataset_name} GD",
        )

        plt.hist(
            cascade_areas,
            bins=bins,
            alpha=0.3,
            label=f"{dataset_name} Cascade",
        )

    plt.xlabel("Bounding Box Area")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, "all_datasets_bbox_area_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    datasets = {
        "PascalVOC": {
            "gd_predictions": "/path/to/benchmark_datasets_predictions_second_round/GroundingDino/PascalVOC/boxthresh_0.01_textthresh_0.2/nms_0.6/val_coco_format.json",
            "cascade_predictions": "/path/to/benchmark_datasets_predictions_second_round/CascadeRcnn/PascalVOC/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/PascalVOC/val_coco_format.json",
        },
        "COCO": {
            "gd_predictions": "/path/to/benchmark_datasets_predictions_second_round/GroundingDino/COCO/boxthresh_0.01_textthresh_0.2/nms_0.6/val_coco_format.json",
            "cascade_predictions": "/path/to/benchmark_datasets_predictions_second_round/CascadeRcnn/COCO/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/COCO/val_coco_format.json",
        },
        "Cityscapes": {
            "gd_predictions": "/path/to/benchmark_datasets_predictions_second_round/GroundingDino/Cityscapes/boxthresh_0.01_textthresh_0.2/nms_0.6/val_coco_format.json",
            "cascade_predictions": "/path/to/benchmark_datasets_predictions_second_round/CascadeRcnn/Cityscapes/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/Cityscapes/val_coco_format.json",
        },
        "KITTI": {
            "gd_predictions": "/path/to/benchmark_datasets_predictions_second_round/GroundingDino/Kitti/boxthresh_0.01_textthresh_0.2/nms_0.6/val_coco_format.json",
            "cascade_predictions": "/path/to/benchmark_datasets_predictions_second_round/CascadeRcnn/Kitti/val_coco_format.json",
            "vgt": "/path/to/benchmark_datasets_vgt/Kitti/val_coco_format.json",
        },
    }

    out_dir = "bbox_area_histograms"

    for dataset_name, paths in datasets.items():
        gd_areas = extract_coco_prediction_areas(paths["gd_predictions"])
        cascade_areas = extract_coco_prediction_areas(paths["cascade_predictions"])
        vgt_areas = extract_legt_areas(paths["vgt"])

        print(dataset_name)
        print(f"  GD predictions:      {len(gd_areas):,}")
        print(f"  Cascade predictions: {len(cascade_areas):,}")
        print(f"  VGT boxes:           {len(vgt_areas):,}")
        print(f"  GD mean area:        {np.mean(gd_areas):.2f}")
        print(f"  Cascade mean area:   {np.mean(cascade_areas):.2f}")
        print(f"  VGT mean area:       {np.mean(vgt_areas):.2f}")

        plot_area_histogram(
            dataset_name=dataset_name,
            gd_areas=gd_areas,
            cascade_areas=cascade_areas,
            vgt_areas=vgt_areas,
            out_dir=out_dir,
            bins=60,
        )

    plot_all_datasets_combined(datasets, out_dir=out_dir, bins=60)


if __name__ == "__main__":
    main()