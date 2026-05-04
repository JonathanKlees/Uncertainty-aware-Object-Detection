import json
import matplotlib.pyplot as plt
import numpy as np


def load_scores_from_coco_predictions(pred_path):
    with open(pred_path, "r") as f:
        predictions = json.load(f)

    scores = []
    for pred in predictions:
        if "score" in pred:
            scores.append(float(pred["score"]))

    return np.array(scores)


def main():
    pred_path = f"/path/to/benchmark_datasets_predictions/GroundingDino"
    
    prediction_files = {
        "PascalVOC": f"{pred_path}/PascalVOC/boxthresh_0.2_textthresh_0.2/nms_0.7/val_coco_format.json",
        "COCO": f"{pred_path}/COCO/boxthresh_0.25_textthresh_0.25/nms_0.6/val_coco_format.json",
        "Cityscapes": f"{pred_path}/Cityscapes/boxthresh_0.2_textthresh_0.2/nms_0.5/val_coco_format.json",
        "KITTI": f"{pred_path}/Kitti/boxthresh_0.2_textthresh_0.2/nms_0.5/val_coco_format.json",
    }

    bins = np.linspace(0.0, 1.0, 51)

    colors = {
        "PascalVOC": "#1f77b4",
        "COCO": "#ff7f0e",
        "Cityscapes": "#2ca02c",
        "KITTI": "#d62728",
    }

    plt.figure(figsize=(8, 5))

    for dataset_name, pred_path in prediction_files.items():
        scores = load_scores_from_coco_predictions(pred_path)

        print(f"{dataset_name}: {len(scores):,} predictions")
        print(f"  min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}, median={np.median(scores):.3f}")

        plt.hist(
            scores,
            bins=bins,
            alpha=0.25,
            histtype="stepfilled",
            label=f"{dataset_name} (#preds = {len(scores):,})",
            color=colors[dataset_name]
        )

        plt.hist(
            scores,
            bins=bins,
            histtype="step",
            linewidth=1.0,
            color=colors[dataset_name]
        )

    plt.xlabel("Grounding DINO prediction score")
    plt.ylabel("Number of predictions")
    plt.title("Score distribution of Grounding DINO predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("gd_score_distribution.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()