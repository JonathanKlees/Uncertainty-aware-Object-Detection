import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import tempfile

def gather_detections(fold):
    # loads per-image detection results from a list of image ids corresponding to json files and converts them into COCO format for evaluation
    detections = []
    for image_id in fold:
        file = f"{image_id}.json"
        if not os.path.exists(os.path.join(pred_dir, file)):
            continue

        with open(os.path.join(pred_dir, file)) as f:
            data = json.load(f)

        # image_id = data["image_id"]
        boxes = data["boxes"]
        scores = data["scores"]
        labels = data["labels"]
        probs = data["probs"]

        for box, score, label, prob in zip(boxes, scores, labels, probs):

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            detections.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [x1, y1, w, h],
                "score": float(score),
                "probs": prob
            })

    return detections

def build_coco_subset(gt_json, image_ids_subset):

    images = [img for img in gt_json["images"]
              if img["id"] in image_ids_subset]

    annotations = [ann for ann in gt_json["annotations"]
                   if ann["image_id"] in image_ids_subset]

    gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": gt_json["categories"]
    }
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    json.dump(gt_dict, open(tmp.name, "w"))
    return tmp.name

def extract_per_image_scores(cocoEval):
    per_image = {}

    for eval_img in cocoEval.evalImgs:
        if eval_img is None:
            continue

        image_id = eval_img["image_id"]
        scores = eval_img["dtScores"]
        matches = eval_img["dtMatches"][0]

        labels = (matches > 0).astype(int)

        if image_id not in per_image:
            per_image[image_id] = {
                "scores": [],
                "labels": []
            }

        per_image[image_id]["scores"].append(scores)
        per_image[image_id]["labels"].append(labels)

    for image_id in per_image:
        per_image[image_id]["scores"] = np.concatenate(per_image[image_id]["scores"])
        per_image[image_id]["labels"] = np.concatenate(per_image[image_id]["labels"])

    return per_image

def flatten_per_image_data(per_image_data):
    scores_all = []
    labels_all = []
    image_ids_all = []

    for img_id, data in per_image_data.items():
        n = len(data["scores"])

        scores_all.append(data["scores"])
        labels_all.append(data["labels"])
        image_ids_all.append(np.full(n, img_id))

    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)
    image_ids_all = np.concatenate(image_ids_all)

    return scores_all, labels_all, image_ids_all


def gather_fold_data(image_ids, per_image_data):
    scores = []
    labels = []

    for img_id in image_ids:
        if img_id not in per_image_data:
            continue

        scores.append(per_image_data[img_id]["scores"])
        labels.append(per_image_data[img_id]["labels"])

    return np.concatenate(scores), np.concatenate(labels)

def compute_ece(scores, labels, n_bins=10):
    # Expected Calibration Error (ECE)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, bins) - 1

    ece = 0.0
    N = len(scores)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        bin_acc = np.mean(labels[mask])
        bin_conf = np.mean(scores[mask])
        ece += (np.sum(mask) / N) * abs(bin_acc - bin_conf)

    return ece

def plot_reliability_diagram(scores, labels, n_bins=10, filename = None):
    # Calibration Plot (Reliability Diagram) with ECE annotation

    ece = compute_ece(scores, labels, n_bins)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, bins) - 1
    bin_means = (bins[:-1] + bins[1:]) / 2

    bin_accs = []
    bin_confs = []

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        bin_accs.append(np.mean(labels[mask]))
        bin_confs.append(np.mean(scores[mask]))

    plt.bar(bin_means[:len(bin_accs)], bin_accs, width=0.1, ec = "slategray", align="center", label="Accuracy")
    plt.plot([0, 1], [0, 1], linestyle="--", lw = 1.5, color="gray")
    plt.xlabel("Mean Confidence")
    plt.ylabel("Mean Accuracy")
    plt.annotate(
        'ECE: %.4f' % ece,
        xy=(0.05, 0.85), xycoords='axes fraction',
        bbox=dict(boxstyle="round", fc="0.9", ec="teal")
    )
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi = 300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate object detection model predictions using Isotonic Regression.")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., coco)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., yolov8l)")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to ground truth annotations of validation set in COCO format")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for k-fold cross-validation (default: 5)")

    parser.add_argument("--visualize", action="store_true", help="Whether to visualize calibration results")
    
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    gt_file = args.gt_file
    k = args.k
    visualize = args.visualize
    
    if visualize:
        os.makedirs("reliability_diagrams", exist_ok=True)

    ##############

    print(f"Calibrating model {model} on dataset {dataset}...")

    with open(gt_file) as f:
        gt_json = json.load(f)

    # folder with per-image json files (required)
    pred_dir = f"../inference/predictions/{model}/{dataset}"

    if dataset == "cityscapes" and model == "gdino": 
        image_ids = [int(fname.split('.')[0]) for fname in os.listdir(pred_dir) if fname.endswith('.json')] # cannot be converted to int
    else:
        image_ids = [int(fname.split('.')[0]) for fname in os.listdir(pred_dir) if fname.endswith('.json')]

    # precompute all detections and corresponding scores and labels for the entire dataset
    all_detections = gather_detections(image_ids)

    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(all_detections)

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    # for calibration, we only care about AP at IoU=0.5, all object sizes and set max dets to 300 so that all detections are evaluated
    cocoEval.params.iouThrs = [0.5]
    cocoEval.params.areaRng = [cocoEval.params.areaRng[0]]
    cocoEval.params.maxDets = [300]

    cocoEval.evaluate()
    cocoEval.accumulate()

    per_image_data = extract_per_image_scores(cocoEval) # {image_id: {"scores": [...], "labels": [...]}, ...} where labels are 1 for TP and 0 for FP based on IoU matching.


    # print(len(per_image_data[1]["scores"]), sum(per_image_data[1]["labels"] > 0)) # example scores and labels for one image

    # print("GT categories:", [c["id"] for c in gt_json["categories"]])
    # print("Example pred labels:", set([d["category_id"] for d in all_detections[:100]]))
    # print("GT image ids example:", gt_json["images"][:5])
    # print("Pred image ids example:", all_detections[:5])
    # print("Pred box example:", all_detections[0]["bbox"])
    # print("GT box example:", gt_json["annotations"][0]["bbox"])

    scores_all, labels_all, image_ids_all = flatten_per_image_data(per_image_data) # all scores, labels and corresponding image ids concatenated into single arrays for easier indexing during cross-validation

    print(f"Total detections: {len(scores_all)}, Total TPs: {sum(labels_all)}, Total FPs: {len(labels_all) - sum(labels_all)}")

    # post-hoc calibration using k-fold cross validation
    np.random.seed(42) # for reproducibility
    np.random.shuffle(image_ids)
    folds = np.array_split(image_ids, k)

    # build fold index per detection
    image_id_to_fold = {}
    for fold_idx, fold in enumerate(folds):
        for img_id in fold:
            image_id_to_fold[img_id] = fold_idx

    fold_indices = np.array([image_id_to_fold[i] for i in image_ids_all])

    all_val_scores = []
    all_val_labels = []
    all_calibrated_val_scores_ps = [] # Platt scaling calibrated scores
    all_calibrated_val_scores_ir = [] # Isotonic regression calibrated scores

    for val_idx in tqdm(list(range(k)), total=k, desc="Cross-validation folds"):
        val_fold = folds[val_idx]
        val_mask = fold_indices == val_idx
        train_mask = ~val_mask

        train_scores = scores_all[train_mask]
        train_labels = labels_all[train_mask]

        val_scores = scores_all[val_mask]
        val_labels = labels_all[val_mask]

        all_val_scores.extend(val_scores.tolist())
        all_val_labels.extend(val_labels.tolist())
        ###########################
        # 1. Platt Scaling (Logistic Regression)
        log_reg = LogisticRegression()
        log_reg.fit(np.array(train_scores).reshape(-1,1), train_labels) # fit to training data

        # store model
        resulting_model = {
            "coef": float(log_reg.coef_[0][0]),
            "intercept": float(log_reg.intercept_[0])
        }
        result_dir = "calibration_models"
        os.makedirs(result_dir, exist_ok=True)
        with open(f"{result_dir}/{model}_{dataset}_platt_scaling_calibration_model_fold_{val_idx}.json", "w") as f:
            json.dump(resulting_model, f)

        # Apply to validation data
        calibrated_val_scores = log_reg.predict_proba(np.array(val_scores).reshape(-1,1))[:,1]
        all_calibrated_val_scores_ps.extend(calibrated_val_scores.tolist())
        ###########################
        # 2. Isotonic Regression

        ir = IsotonicRegression(out_of_bounds='clip')  # clip scores outside training range
        ir.fit(np.array(train_scores).reshape(-1,1), train_labels)

        # store model
        resulting_model = {
            "X_thresholds": ir.X_thresholds_.tolist(),
            "y_thresholds": ir.y_thresholds_.tolist()
        }
        with open(f"{result_dir}/{model}_{dataset}_isotonic_regression_calibration_model_fold_{val_idx}.json", "w") as f:
            json.dump(resulting_model, f)

        # Apply to validation data
        IR_calibrated_scores = ir.predict(np.array(val_scores).reshape(-1,1))
        all_calibrated_val_scores_ir.extend(IR_calibrated_scores.tolist())
        ###########################
        # 3. Store calibrated probabilities

        for filename in val_fold:

            with open(os.path.join(pred_dir, f"{filename}.json")) as f:
                data = json.load(f)

            boxes = data["boxes"]
            scores = data["scores"]
            labels = data["labels"]
            probs = data["probs"]

            probs_array = np.array(probs)  # shape: (num_detections, num_classes)

            if probs_array.size == 0: # occured only for YOLOv8 on Cityscapes, some image with zero detections
                print(f"Warning: No detections for image {filename} in fold {val_idx}, skipping calibration for this image.")
                # nothing to calibrate
                data["calibrated_probs_ir"] = []
                data["calibrated_probs_ps"] = []
                
                with open(f"{pred_dir}/{filename}.json", "w") as f:
                    json.dump(data, f)
                
                continue

            # flatten
            flat_probs = probs_array.reshape(-1, 1)

            # apply calibration
            flat_ir = ir.predict(flat_probs)
            flat_ps = log_reg.predict_proba(flat_probs)[:, 1]

            # reshape back
            calibrated_probs_ir = flat_ir.reshape(probs_array.shape).tolist()
            calibrated_probs_ps = flat_ps.reshape(probs_array.shape).tolist()
            
            data["calibrated_probs_ir"] = calibrated_probs_ir
            data["calibrated_probs_ps"] = calibrated_probs_ps

            # store updated detection with calibrated probabilities
            with open(f"{pred_dir}/{filename}.json", "w") as f:
                json.dump(data, f)

    # Compute ECE for aggregated validation data across all folds before and after calibration, 
    # and store results in a csv file for comparison between methods and datasets. 
    # Also generate reliability diagrams for visual comparison of calibration performance.

    all_val_scores = np.array(all_val_scores)
    all_val_labels = np.array(all_val_labels)
    all_calibrated_val_scores_ps = np.array(all_calibrated_val_scores_ps)
    all_calibrated_val_scores_ir = np.array(all_calibrated_val_scores_ir)

    # Compare ECE before and after calibration (store below)
    ece_before = compute_ece(all_val_scores, all_val_labels, n_bins=10)
    ece_calibrated_PS = compute_ece(all_calibrated_val_scores_ps, all_val_labels, n_bins=10)

    if visualize:
        plot_reliability_diagram(all_val_scores, all_val_labels, n_bins=10, filename=f"reliability_diagrams/{model}_{dataset}_before_calibration.pdf")

        plot_reliability_diagram(all_calibrated_val_scores_ps, all_val_labels, n_bins=10, filename=f"reliability_diagrams/{model}_{dataset}_platt_scaling.pdf")


    # Compare ECE after calibration with Platt scaling results
    ece_calibrated_IR = compute_ece(all_calibrated_val_scores_ir, all_val_labels, n_bins=10)

    if visualize:
         plot_reliability_diagram(all_calibrated_val_scores_ir, all_val_labels, n_bins=10, filename=f"reliability_diagrams/{model}_{dataset}_isotonic_regression.pdf")
   
    ###########################
    # 4. Store ECE results for both methods in a csv file
    result_csv = "calibration_results.csv"

    new_row = {
        "dataset": dataset,
        "model": model,
        "ECE_raw": ece_before,
        "ECE_platt_scaling": ece_calibrated_PS,
        "ECE_isotonic_regression": ece_calibrated_IR
    }

    if os.path.exists(result_csv):
        results = pd.read_csv(result_csv)
        results.loc[len(results)] = new_row
    else:
        results = pd.DataFrame([new_row])

    results.to_csv(result_csv, index=False)