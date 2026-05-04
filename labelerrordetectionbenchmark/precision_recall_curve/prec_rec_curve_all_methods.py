import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


IOU_THRESH = 0.1


DATASET_NAME = "PascalVOC"
PRED_NET = "CascadeRcnn"

COLORS = {
    "score": "#1f77b4",      # blue
    "loss": "#d62728",       # red
    "objectlab": "#2ca02c",  # green
    "metadetect": "#ff7f0e", # orange
    "naive": "#000000",      # black
}

GT_CONFIGS = {
    "less_restrictive": f"/path/to/benchmark_datasets_legt_second_round_variant1/{DATASET_NAME}/val/val_coco_format.json",
    "stricter": f"/path/to/benchmark_datasets_legt_second_round_variant2/{DATASET_NAME}/val/val_coco_format.json",
}

METHOD_CONFIGS = {
    "Score baseline": {
        "path": f"/path/to/benchmark_datasets_labelerrorproposals_second_round/{PRED_NET}/naive_baseline/{DATASET_NAME}/iou_thresh_0.5/score_all_val_coco_format.json",
        "color": COLORS["score"],
        "is_naive": False,
    },
    "Loss inspection": {
        "path": f"/path/to/benchmark_datasets_labelerrorproposals_second_round/CascadeRcnn/lossmethod/{DATASET_NAME}/score_thresh_0.01_val_coco_format.json",
        "color": COLORS["loss"],
        "is_naive": False,
    },
    "ObjectLab": {
        "path": f"/path/to/benchmark_datasets_labelerrorproposals_second_round/{PRED_NET}/objectlab/naive_iou_0.5/{DATASET_NAME}/val/1.0_val_coco_format.json",
        "color": COLORS["objectlab"],
        "is_naive": False,
    },
    # "MetaDetect": {
    #     "path": f"/path/to/benchmark_datasets_labelerrorproposals_second_round/{PRED_NET}/metadetect/{DATASET_NAME}/iou_thresh_0.5/val_score_thresh_0.01_coco_format.json",
    #     "color": COLORS["metadetect"],
    #     "is_naive": False,
    # },
    "Naive baseline": {
        "path": f"/path/to/benchmark_datasets_labelerrorproposals_second_round/{PRED_NET}/naive_baseline/{DATASET_NAME}/iou_thresh_0.5/naive_val_coco_format.json",
        "color": COLORS["naive"],
        "is_naive": True,
    },
}

OUTPUT_DIR = Path(f"pr_results/{PRED_NET}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON_PATH = OUTPUT_DIR /  f"{DATASET_NAME}_pr_results.json"



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_gt_annotations(gt_path):
    data = load_json(gt_path)

    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]

    if isinstance(data, list):
        return data

    raise ValueError(f"Unknown GT format: {gt_path}")


def load_predictions(pred_path):
    data = load_json(pred_path)

    if not isinstance(data, list):
        raise ValueError(f"Prediction file must be a list: {pred_path}")

    return data



def compute_iou_xywh(box1, box2):
    x1, y1, w1, h1 = map(float, box1)
    x2, y2, w2, h2 = map(float, box2)

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h

    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def group_gts_by_image(gts):
    gt_dict = defaultdict(list)

    for gt in gts:
        gt_dict[str(gt["image_id"])].append({
            "bbox": gt["bbox"],
            "matched": False,
        })

    return gt_dict



def compute_pr_curve(preds, gts, iou_thresh=0.1):
    gt_dict = group_gts_by_image(gts)
    total_gt = sum(len(v) for v in gt_dict.values())

    if total_gt == 0:
        raise ValueError("No GT label errors found.")

    valid_preds = []
    for p in preds:
        if "score" not in p:
            continue
        if "bbox" not in p:
            continue
        valid_preds.append(p)

    preds = sorted(valid_preds, key=lambda x: float(x["score"]), reverse=True)

    tp_list = []
    fp_list = []
    scores = []

    for pred in preds:
        image_id = str(pred["image_id"])
        pred_box = pred["bbox"]
        pred_score = float(pred["score"])

        gt_boxes = gt_dict.get(image_id, [])

        best_iou = 0.0
        best_gt = None

        for gt in gt_boxes:
            if gt["matched"]:
                continue

            iou = compute_iou_xywh(pred_box, gt["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= iou_thresh and best_gt is not None:
            tp_list.append(1)
            fp_list.append(0)
            best_gt["matched"] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

        scores.append(pred_score)

    if len(tp_list) == 0:
        return np.array([]), np.array([]), np.array([])

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / total_gt
    scores = np.asarray(scores)

    return precision, recall, scores


def compute_naive_point(preds, gts, iou_thresh=0.1):
    gt_dict = group_gts_by_image(gts)
    total_gt = sum(len(v) for v in gt_dict.values())

    tp = 0
    fp = 0

    for pred in preds:
        image_id = str(pred["image_id"])
        pred_box = pred["bbox"]

        gt_boxes = gt_dict.get(image_id, [])

        best_iou = 0.0
        best_gt = None

        for gt in gt_boxes:
            if gt["matched"]:
                continue

            iou = compute_iou_xywh(pred_box, gt["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_iou >= iou_thresh and best_gt is not None:
            tp += 1
            best_gt["matched"] = True
        else:
            fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "num_predictions": int(len(preds)),
        "num_gt": int(total_gt),
    }


def compute_ap(precision, recall):
    precision = np.asarray(precision)
    recall = np.asarray(recall)

    if len(precision) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([precision[0]], precision))

    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
    return float(ap)


def compute_best_f1(precision, recall, scores):
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    scores = np.asarray(scores)

    if len(precision) == 0:
        return {
            "best_f1": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_threshold": None,
            "best_index": None,
        }

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1))

    return {
        "best_f1": float(f1[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_threshold": float(scores[best_idx]),
        "best_index": best_idx,
    }



def plot_all_methods(dataset_name, gt_name, method_results, output_path):
    plt.figure(figsize=(7, 6))

    line_styles = {
        "Score baseline": "-",
        "Loss inspection": "-",
        "ObjectLab": "-",
        "MetaDetect": "-",
    }

    line_alpha = {
        "Score baseline": 1.0,
        "Loss inspection": 1.0,
        "ObjectLab": 1.0,
        "MetaDetect": 0.85,
    }

    for method_name, res in method_results.items():
        color = res.get("color", None)

        if res["is_naive"]:
            plt.scatter(
                res["recall"],
                res["precision"],
                marker="o",
                s=75,
                color=color if color is not None else "black",
                edgecolors="white",
                linewidths=1.2,
                label=method_name,
                zorder=10,
            )
            continue

        precision = np.asarray(res["precision"])
        recall = np.asarray(res["recall"])

        plt.plot(
            recall,
            precision,
            linewidth=2.4,
            linestyle=line_styles.get(method_name, "-"),
            alpha=line_alpha.get(method_name, 1.0),
            color=color,
            label=method_name,
        )

        plt.scatter(
            res["best_recall"],
            res["best_precision"],
            marker="x",
            color=color,
            s=55,
            linewidths=2,
            zorder=8,
        )
        
    if gt_name == "less_restrictive":
        gt_name = "less restrictive"

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    #plt.title(f"Precision–Recall Curve on {dataset_name} ({gt_name} LEGT)")
    plt.grid(True, alpha=0.25)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_dataset_all_methods(
    dataset_name,
    gt_configs,
    method_configs,
    output_dir,
    iou_thresh=0.1,
):
    all_results = {}

    for gt_name, gt_path in gt_configs.items():
        print(f"\n=== Evaluating GT config: {gt_name} ===")

        gts = load_gt_annotations(gt_path)
        gt_results = {}

        for method_name, method_cfg in method_configs.items():
            pred_path = method_cfg["path"]
            is_naive = method_cfg.get("is_naive", False)
            method_color = method_cfg.get("color", None)

            print(f"  Method: {method_name}")

            preds = load_predictions(pred_path)

            if is_naive:
                naive_res = compute_naive_point(
                    preds=preds,
                    gts=gts,
                    iou_thresh=iou_thresh,
                )

                gt_results[method_name] = {
                    "is_naive": True,
                    "color": method_color,
                    "ap": None,
                    "best_f1": naive_res["f1"],
                    "best_precision": naive_res["precision"],
                    "best_recall": naive_res["recall"],
                    "best_threshold": None,
                    "precision": naive_res["precision"],
                    "recall": naive_res["recall"],
                    "f1": naive_res["f1"],
                    "tp": naive_res["tp"],
                    "fp": naive_res["fp"],
                    "num_predictions": naive_res["num_predictions"],
                    "num_gt": naive_res["num_gt"],
                }

                print(
                    f"    Naive point: "
                    f"F1={naive_res['f1']:.4f}, "
                    f"P={naive_res['precision']:.4f}, "
                    f"R={naive_res['recall']:.4f}"
                )

            else:
                precision, recall, scores = compute_pr_curve(
                    preds=preds,
                    gts=gts,
                    iou_thresh=iou_thresh,
                )

                ap = compute_ap(precision, recall)
                best_f1 = compute_best_f1(precision, recall, scores)

                gt_results[method_name] = {
                    "is_naive": False,
                    "color": method_color,
                    "ap": ap,
                    "best_f1": best_f1["best_f1"],
                    "best_precision": best_f1["best_precision"],
                    "best_recall": best_f1["best_recall"],
                    "best_threshold": best_f1["best_threshold"],
                    "best_index": best_f1["best_index"],
                    "num_predictions": len(preds),
                    "num_gt": len(gts),
                    "precision": precision,
                    "recall": recall,
                    "scores": scores,
                }

                print(
                    f"    AP={ap:.4f}, "
                    f"Best F1={best_f1['best_f1']:.4f}, "
                    f"P={best_f1['best_precision']:.4f}, "
                    f"R={best_f1['best_recall']:.4f}, "
                    f"Thresh={best_f1['best_threshold']:.6f}"
                )

        plot_path = output_dir / f"{dataset_name}_{gt_name}_all_methods_pr.png"

        plot_all_methods(
            dataset_name=dataset_name,
            gt_name=gt_name,
            method_results=gt_results,
            output_path=plot_path,
        )

        print(f"Saved plot to: {plot_path}")

        all_results[gt_name] = gt_results

    return all_results


def convert_results_for_json(results):
    json_ready = {}

    for gt_name, gt_results in results.items():
        json_ready[gt_name] = {}

        for method_name, res in gt_results.items():
            clean_res = {}

            for key, value in res.items():
                if isinstance(value, np.ndarray):
                    clean_res[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    clean_res[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    clean_res[key] = int(value)
                else:
                    clean_res[key] = value

            json_ready[gt_name][method_name] = clean_res

    return json_ready


if __name__ == "__main__":
    results = evaluate_dataset_all_methods(
        dataset_name=DATASET_NAME,
        gt_configs=GT_CONFIGS,
        method_configs=METHOD_CONFIGS,
        output_dir=OUTPUT_DIR,
        iou_thresh=IOU_THRESH,
    )

    json_ready_results = convert_results_for_json(results)

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(json_ready_results, f, indent=4)

    print(f"\nSaved results JSON to: {RESULTS_JSON_PATH}")