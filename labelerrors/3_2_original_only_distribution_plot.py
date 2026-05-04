import os
import json
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


def compute_box_area(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def load_original_only_areas_from_json_dir(json_dir, normalize=False):
    json_files = sorted(glob(os.path.join(json_dir, "*.json")))
    areas = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        img_w = data.get("width", None)
        img_h = data.get("height", None)

        for err in data.get("errors", []):
            if err.get("type") != "original_only":
                continue

            bbox = err.get("original_box", {}).get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            area = compute_box_area(bbox)
            if area <= 0:
                continue

            if normalize:
                if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
                    continue
                area = area / float(img_w * img_h)

            areas.append(area)

    return np.array(areas, dtype=float)


def load_original_only_areas_from_multiple_dirs(json_dirs, normalize=False):
    all_areas = []
    for json_dir in json_dirs:
        vals = load_original_only_areas_from_json_dir(json_dir, normalize=normalize)
        if len(vals) > 0:
            all_areas.append(vals)

    if len(all_areas) == 0:
        return np.array([], dtype=float)

    return np.concatenate(all_areas)


def plot_original_only_area_curves_filled(
    dataset_dirs,
    normalize=False,
    bins=50,
    density=False,
    save_path=None
):
    dataset_names = ["KITTI", "Cityscapes", "PascalVOC", "COCO"]

    colors = {
        "KITTI": "#1f77b4",
        "Cityscapes": "#ff7f0e",
        "PascalVOC": "#2ca02c",
        "COCO": "#d62728",
    }

    dataset_values = {}
    for name in dataset_names:
        values = load_original_only_areas_from_multiple_dirs(
            dataset_dirs[name],
            normalize=normalize
        )
        values = values[values > 0]
        dataset_values[name] = values
        print(f"{name}: {len(values)} original-only boxes")

    combined = np.concatenate([v for v in dataset_values.values() if len(v) > 0])

    bin_edges = np.logspace(
        np.log10(combined.min()),
        np.log10(combined.max()),
        bins
    )

    plt.figure(figsize=(9, 5.5))

    for name in dataset_names:
        values = dataset_values[name]
        if len(values) == 0:
            continue

        counts, edges = np.histogram(values, bins=bin_edges, density=density)
        centers = np.sqrt(edges[:-1] * edges[1:])

        # Linie
        plt.plot(
            centers,
            counts,
            color=colors[name],
            linewidth=2.5,
            label=name
        )

        # leichte Füllung (KEY PART)
        plt.fill_between(
            centers,
            counts,
            alpha=0.15,              # sehr wichtig: niedrig halten!
            color=colors[name]
        )


    plt.xscale("log")
    plt.xlabel("Normalized Bounding Box Area" if normalize else "Bounding Box Area")
    plt.ylabel("Density" if density else "Frequency")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    
if __name__ == '__main__':
    dataset_dirs = {
        "KITTI": [
            "/path/to/benchmark_datasets_legt/Kitti/train/json",
            "/path/to/benchmark_datasets_legt/Kitti/val/json",
        ],
        "Cityscapes": [
            "/path/to/benchmark_datasets_legt/Cityscapes/train/json",
            "/path/to/benchmark_datasets_legt/Cityscapes/val/json",
        ],
        "PascalVOC": [
            "/path/to/benchmark_datasets_legt/PascalVOC/train/json",
            "/path/to/benchmark_datasets_legt/PascalVOC/val/json",
        ],
        "COCO": [
            "/path/to/benchmark_datasets_legt/COCO/val/json",
        ],
    }

    plot_original_only_area_curves_filled(
        dataset_dirs=dataset_dirs,
        normalize=False,
        bins=50,
        density=False,
        save_path="/path/to/benchmark_datasets_legt/original_only_comparison/original_only_size_distribution.png"
    )

