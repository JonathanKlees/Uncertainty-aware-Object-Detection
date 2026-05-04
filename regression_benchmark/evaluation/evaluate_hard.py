import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import defaultdict
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from torchvision.ops import box_iou

def mAP_evaluation(dataset, model, gt_name, gt_file, result_csv, soft_labels_available, soft_label_thresh):
    """Evaluate the model's predictions against the ground truth using COCO evaluation metrics.
    Computes mean average precision (mAP) and average recall (AR) at various IoU thresholds and object sizes.
    Hard label comparison.
    Saves the results to a CSV file for later analysis."""

    pred_dir = f"../inference/predictions/{model}/{dataset}"   # folder with per-image json files (required)
    
    detections = []

    for file in os.listdir(pred_dir):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(pred_dir, file)) as f:
            data = json.load(f)

        image_id = file.split(".")[0] # data["image_id"]
        boxes = data["boxes"]
        scores = data["scores"]
        labels = data["labels"]

        offset = 0 # if model uses 1-based indexing for labels, set to 1, else 0
        if model == "frcnn" and dataset != "coco": # faster rcnn uses 1-based indexing for predictions, zero is reserved for background class
            offset = -1

        for box, score, label in zip(boxes, scores, labels):

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            detections.append({
                "image_id": int(image_id),
                "category_id": int(label) + offset,
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })

    if soft_labels_available: # have to convert to hard labels
        with open(gt_file) as f:
            data = json.load(f)

        gt_file = convert_to_hard_labels(data, dataset, soft_label_thresh)

    # load COCO
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(detections)

    # evaluate
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Store to CSV

    metrics = [
        "AP","AP50","AP75",
        "AP_small","AP_medium","AP_large",
        "AR_1","AR_10","AR_100",
        "AR_small","AR_medium","AR_large"
    ]

    new_row = {
        "dataset": dataset,
        "model": model,
        "gt": gt_name
    }

    new_row.update(dict(zip(metrics, cocoEval.stats.tolist())))

    if os.path.exists(f"results/{result_csv}"):
        results = pd.read_csv(f"results/{result_csv}")
        results.loc[len(results)] = new_row
    else:
        results = pd.DataFrame([new_row])

    results.to_csv(f"results/{result_csv}", index=False)

def convert_to_hard_labels(dataset, dataset_name, prob_thresh = 0.5):
    """ converts annotations with soft labels to hard labels based on a prob. thresh.
    if the maximal probability of the soft label vector exceeds the thresh, an annotation is formed."""

    filename_to_id = {
            img["file_name"]: img["id"]
            for img in dataset["images"]
        }
    
    if dataset_name == "coco": # coco processing: map to original coco class ids

        id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                        41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    else:
        id_map = {i : i for i in range(len(dataset["categories"]))} # identity mapping if not coco
    

    # initialize new dataset with hard labels in coco format
    coco = {
        "images": dataset["images"],
        "annotations": [], # initialize empty annotations list
        "categories": dataset["categories"]
    }

    annotation_id = 1

    i = 0

    for img_name, annotations in dataset["objects"].items():

        if dataset_name in ["cityscapes", "kitti"]: # cityscapes and kitti use file names without extension, so we add it back here
                img_name = img_name + ".png"

        image_id = filename_to_id[img_name]

        coco["images"].append({
            "id": image_id,
            "file_name": img_name
        })

        for annot in annotations:

            bbox = annot["bbox"]
            soft_label = annot["soft_label"][:-1] # ignore background class (cant_solve)

            cls = int(np.argmax(soft_label))
            prob = soft_label[cls]

            if prob > prob_thresh:

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": id_map[cls],
                    "bbox": [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ],
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "iscrowd": 0
                })

                annotation_id += 1
            
            i+=1
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco, f)
        temp_path = f.name
    return temp_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Evaluation Protocol", add_help=True)
    # load model
    parser.add_argument("--model", "-m", type=str,
                        required=True, help="model name (e.g., 'yolo', 'gdino')")
    parser.add_argument(
        "--dataset", "-d", type=str, required=True, help="dataset name (e.g., 'coco')"
    )

    parser.add_argument("--gt_name", type=str, help="name of considered ground truth (e.g., coco_orig_gt)")

    parser.add_argument("--gt_file", type=str, help="path to COCO-format ground truth JSON file")

    parser.add_argument("--soft_labels_available", action="store_true", help="Soft Labels in Dataset")

    parser.add_argument("--soft_label_thresh", type = float, default=0.0, help="Threshold to convert soft label annotations to hard labels.")

    parser.add_argument("--mAP_result_csv", type=str, default="mAP_results.csv", help="CSV file to store mAP evaluation results")


    args = parser.parse_args()

    # Compute mean average precision with hard label comparison, stores results in a CSV file for later analysis. 
    # If soft labels are available, they are converted to hard labels based on the provided threshold before evaluation.

    mAP_evaluation(args.dataset, args.model, args.gt_name, args.gt_file, args.mAP_result_csv, args.soft_labels_available, args.soft_label_thresh)