import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from collections import defaultdict

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Utils
# -----------------------------
def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def compute_iou_matrix(boxes1, boxes2):
    # boxes: (N, 4) format [x1, y1, x2, y2]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-8)


def build_coco_mapping(device="cuda"):
    # for COCO, we need to map from our 0-79 category IDs to the original COCO category IDs. this is the mapping for it:
    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                    41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    inverse_map = {v: k for k, v in id_map.items()} # map model output idx -> GT idx

    model_indices = torch.tensor(list(inverse_map.keys()), device=device)
    gt_indices = torch.tensor(list(inverse_map.values()), device=device)

    return model_indices, gt_indices

def gather_predictions(dataset, model):
    """Gather per-image predictions and their associated probabilities from the saved JSON files.
    Returns a dataframe containing predicted boxes with image_ids, scores, labels, and probabilities."""

    pred_dir = f"../inference/predictions/{model}/{dataset}"   # folder with per-image json files (required)
  
    # construct detections list in COCO format from per-image json files
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
        # probs = data["probs"]
        probs = data["calibrated_probs_ir"] # use calibrated probabilities (isotonic regression) for evaluation

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

def gather_annotations(dataset, gt_file):
    """Gather ground truth annotations from the COCO-format JSON file.
    Returns a dataframe containing GT boxes with image_ids, category_ids, and bounding boxes."""

    with open(gt_file) as f:
        data = json.load(f)

    filename_to_id = {
        img["file_name"]: img["id"]
        for img in data["images"] # information on images from the original COCO dataset
    }

    if dataset == "coco": # for COCO, we need to map from file names to image IDs and from our 0-79 category IDs to the original COCO category IDs
        
        id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                        41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    else: # for other datasets, define identity mappings

        id_map = {i: i for i in range(len(data["categories"]))} # identity mapping

    objects = data["objects"]

    annotations = []

    append = annotations.append
    filename_lookup = filename_to_id
    category_lookup = id_map

    annotation_id = 1

    for img_name, anns in objects.items():

        if dataset in ["kitti", "cityscapes"]:
            img_name = img_name + ".png" # for KITTI and Cityscapes, image names in annotations don't have .png extension, but they do in the filename_to_id mapping, so we need to add it back
        
        image_id = filename_lookup[img_name]

        for ann in anns:

            bbox = ann["bbox"]
            soft_label = ann["postprocessed_soft_label"] # ann["soft_label"]

            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            # leave out annotations with cant solve as most likely category
            if int(np.argmax(soft_label)) == len(soft_label) - 1: # last category is "cant solve"
                continue

            append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_lookup[int(np.argmax(soft_label))],
                "soft_label": soft_label,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })

            annotation_id += 1

    return annotations

def prepare_image_data(detections, annotations, dataset, model, mapping, device="cuda"):
    pred_by_image = defaultdict(list)
    gt_by_image = defaultdict(list)

    for det in detections:
        pred_by_image[det["image_id"]].append(det)

    for ann in annotations:
        gt_by_image[ann["image_id"]].append(ann)

    image_data = {}

    for image_id in pred_by_image.keys():

        preds = pred_by_image[image_id]
        gts = gt_by_image.get(image_id, [])

        if len(preds) == 0:
            continue

        pred_boxes = torch.tensor([
            [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            for b in [p["bbox"] for p in preds]
        ], dtype=torch.float32, device=device)

        pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        
        pred_scores = torch.tensor(
            [p["score"] for p in preds],
            dtype=torch.float32,
            device=device
        )

        raw_probs = [p["probs"] for p in preds]

        # convert once to tensor
        raw_probs = torch.tensor(raw_probs, dtype=torch.float32, device=device)

        # precompute aligned probabilities
        probs = compute_probs(
            pred_probs=raw_probs,
            dataset=dataset,   # or pass dataset if needed
            model=model,       # not used anyway in your function
            mapping=mapping,
            device=device
        )

        if len(gts) > 0:
            gt_boxes = torch.tensor(
                [xywh_to_xyxy(g["bbox"]) for g in gts],
                dtype=torch.float32,
                device=device
            )

            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

            gt_labels = torch.tensor(
                [g["soft_label"] for g in gts],
                dtype=torch.float32,
                device=device
            )
        else:
            gt_boxes = torch.empty((0, 4), device=device)
            gt_labels = torch.empty((0, 81), device=device)
            gt_areas = torch.empty((0,), device=device)

        # compute IoU ONCE
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

        image_data[image_id] = {
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "probs": probs, # already processed for efficiency
            "pred_areas": pred_areas,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_areas": gt_areas,
            "iou_matrix": iou_matrix
        }

    return image_data

def chunked_image_data(image_data, chunk_size=64):
    items = list(image_data.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i+chunk_size])


def process_chunk(
    chunk,
    mapping,
    dataset,
    model,
    area_range,
    iou_thresh,
    device,
    scores_to_sort,
    similarity_measures
):
    all_scores = {k: [] for k in scores_to_sort}
    all_sims = {k: [] for k in similarity_measures}
    all_tp = []
    total_gt = 0

    for image_id, data in chunk.items():

        matched_pred, dt_ignore, gt_ignore = coco_style_matching_fast(
            data["iou_matrix"],
            data["pred_scores"],
            data["pred_areas"],
            data["gt_areas"],
            iou_thresh,
            area_range
        )

        probs = data["probs"]

        gt_full = torch.zeros_like(probs)
        gt_full[:, -1] = 1.0

        matched_mask = matched_pred >= 0
        if matched_mask.any():
            gt_full[matched_mask] = data["gt_labels"][matched_pred[matched_mask]]

        valid_mask = ~dt_ignore
        probs = probs[valid_mask]
        gt_full = gt_full[valid_mask]
        matched_mask = matched_mask[valid_mask]

        metrics = compute_metrics(probs, gt_full)
        metrics_matched = {k: v[matched_mask] for k, v in metrics.items()}  # only matched predictions


        for score in scores_to_sort:
            all_scores[score].append(metrics[score])
        for sim in similarity_measures:
            all_sims[sim].append(metrics[sim])

        bbox_tp = (matched_pred >= 0) & (~dt_ignore)
        all_tp.append(bbox_tp[valid_mask])

        total_gt += (~gt_ignore).sum().item()

    return all_scores, all_sims, all_tp, total_gt, metrics, metrics_matched

def coco_style_matching_fast(
    iou_matrix,
    scores,
    pred_areas,
    gt_areas,
    iou_thresh,
    area_range
):
    area_min, area_max = area_range

    device = iou_matrix.device
    Np, Ng = iou_matrix.shape

    if Ng == 0:
        matched_pred = torch.full((Np,), -1, dtype=torch.long, device=device)
        dt_ignore = (
            (pred_areas < area_min) | (pred_areas > area_max)
        )
        gt_ignore = torch.zeros((0,), dtype=torch.bool, device=device)
        return matched_pred, dt_ignore, gt_ignore

    # --- GT ignore ---
    gt_ignore = (gt_areas < area_min) | (gt_areas > area_max)

    # -------------------------------------------------
    # STEP 1: mask IoU below threshold
    # -------------------------------------------------
    valid_iou = iou_matrix >= iou_thresh

    # -------------------------------------------------
    # STEP 2: prefer non-ignored GTs
    # -------------------------------------------------
    iou_valid = iou_matrix.clone()

    # first pass: non-ignored GTs
    iou_valid[:, gt_ignore] = -1

    best_iou, best_gt = iou_valid.max(dim=1)

    # fallback: if no valid GT found → allow ignored GT
    fallback_mask = best_iou < iou_thresh

    if fallback_mask.any():
        iou_fallback = iou_matrix[fallback_mask]
        best_iou_fb, best_gt_fb = iou_fallback.max(dim=1)

        use_fb = best_iou_fb >= iou_thresh
        best_gt[fallback_mask] = torch.where(
            use_fb,
            best_gt_fb,
            best_gt[fallback_mask]
        )
        best_iou[fallback_mask] = torch.where(
            use_fb,
            best_iou_fb,
            best_iou[fallback_mask]
        )

    # -------------------------------------------------
    # STEP 3: enforce 1-to-1 matching (vectorized)
    # -------------------------------------------------
    valid = best_iou >= iou_thresh

    pred_idx = torch.arange(Np, device=device)[valid]
    gt_idx = best_gt[valid]
    pred_scores = scores[valid]

    # sort by score descending
    order = torch.argsort(pred_scores, descending=True)
    pred_idx = pred_idx[order]
    gt_idx = gt_idx[order]

    # unique GT assignment (small loop, but cheap!)
    matched_pred = torch.full((Np,), -1, dtype=torch.long, device=device)
    used_gt = torch.zeros(Ng, dtype=torch.bool, device=device)

    for i in range(len(pred_idx)):
        p = pred_idx[i]
        g = gt_idx[i]

        if not used_gt[g]:
            matched_pred[p] = g
            used_gt[g] = True

    # -------------------------------------------------
    # STEP 4: detection ignore (same as your code)
    # -------------------------------------------------
    dt_ignore = torch.zeros(Np, dtype=torch.bool, device=device)

    matched_mask = matched_pred >= 0

    # matched to ignored GT
    dt_ignore[matched_mask] = gt_ignore[matched_pred[matched_mask]]

    # unmatched + out of area
    unmatched = ~matched_mask
    dt_ignore[unmatched] = (
        (pred_areas[unmatched] < area_min) |
        (pred_areas[unmatched] > area_max)
    )

    return matched_pred, dt_ignore, gt_ignore


# -----------------------------
# Probability processing & Metrics
# -----------------------------

def compute_probs(pred_probs, dataset, model, mapping, device="cuda"):
    """ 
    predicted probabilities of COCO models are often for 91 classes where non-zero probs occur only for 80 classes that match COCO

    some models such as gdino may not output a softmax distribution directly

    this function generates a prob. distribution over the 80 COCO classes + background from the model output probabilities

    the logic may depend on the model

    Align model output probs to GT classes, add background, and normalize.

    Args:
        pred_probs: np.array of shape (model_output_dim,)
        model: str, name of the model (e.g., "yolo", "gdino") to determine how to interpret the output probabilities
    Returns:
        aligned_probs: np.array of shape (gt_dim+1,), last entry is background
    """

    if not torch.is_tensor(pred_probs):
        pred_probs = torch.tensor(pred_probs, dtype=torch.float32, device=device)
    else:
        pred_probs = pred_probs.to(device=device, dtype=torch.float32)

    if dataset == "coco" and "gdino" in model.lower():
        model_indices, gt_indices = mapping

        N = pred_probs.shape[0]
        gt_dim = 80

        aligned = torch.zeros((N, gt_dim), device=device)

        # vectorized mapping
        aligned[:, gt_indices] = pred_probs[:, model_indices]

    else:
        aligned = pred_probs  # fallback if not COCO

    # add background as the product of (1 - prob) across all classes (assuming independence)
    if not model == "frcnn": # Faster RCNN already outputs a background class, so skip adding background for it
        bg = torch.prod(1.0 - aligned, dim=1, keepdim=True)

        probs = torch.cat([aligned, bg], dim=1)
    
    else:
        probs = aligned
    
    probs = probs / probs.sum(dim=1, keepdim=True)

    return probs



def compute_metrics(probs, gt, eps = 1e-8):
    # cosine
    cos = torch.nn.functional.cosine_similarity(probs, gt, dim=1)

    # TV
    tv = 0.5 * torch.sum(torch.abs(probs - gt), dim=1)
    tv_sim = 1.0 - tv

    # Jensen-Shannon divergence
    m = 0.5 * (probs + gt)
    kl_pm = torch.sum(probs * torch.log((probs + eps) / (m + eps)), dim=1)
    kl_qm = torch.sum(gt * torch.log((gt + eps) / (m + eps)), dim=1)
    js_div = 0.5 * (kl_pm + kl_qm)
    js_dist = torch.sqrt(js_div)
    
    js_sim = 1.0 - js_dist

    # entropy score
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)

    # max prob
    max_prob, _ = torch.max(probs, dim=1)

    return {
        "cosine": cos,
        "tv": tv_sim,
        "js": js_sim,
        "cosine_dist": 1.0 - cos,
        "TVD": tv, # total variation distance
        "JSD": js_dist, # Jensen-Shannon distance
        "MAE": torch.mean(torch.abs(probs - gt), dim=1), # mean absolute error
        "MSE": torch.mean((probs - gt) ** 2, dim=1), # mean squared error
        "KL": torch.sum(gt * torch.log((gt + eps) / (probs + eps)), dim=1), # KL divergence from GT to prediction
        "entropy": entropy,
        "maxprob": max_prob
    }


def compute_AUSRC(scores, similarities, bbox_tp, total_gt):
    order = torch.argsort(scores, descending=True)
    sims = similarities[order]
    tp = bbox_tp[order].float()

    # cumulative TP → recall
    cum_tp = torch.cumsum(tp, dim=0)
    recall = cum_tp / total_gt

    cum_mean_sim = torch.cumsum(sims, dim=0) / torch.arange(
        1, len(sims)+1, device=sims.device
    )

    recall_levels = torch.linspace(0, 1, 101, device=sims.device)

    interp_sim = torch.zeros_like(recall_levels)

    for i, r in enumerate(recall_levels):
        mask = recall >= r
        interp_sim[i] = torch.max(cum_mean_sim[mask]) if mask.any() else 0.0

    return interp_sim.mean().item()


# -----------------------------
# Main evaluation
# -----------------------------
def AUSRC_evaluation(
    dataset,
    model,
    detections,
    annotations,
    iou_thresholds=np.linspace(0.5, 0.95, 10),
    similarity_measures=["tv"],
    metrics = ["MAE", "MSE", "KL", "JSD", "TVD", "cosine_dist"],
    scores_to_sort=["maxprob"],
    device="cuda",
    area_ranges={"all": (0, 1e10), "small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e10)},
    num_workers=8,
    chunk_size=64,
):

    if dataset == "coco":
        mapping = build_coco_mapping(device=device)
    else:
        mapping = (
            torch.tensor(np.arange(len(detections[0]["probs"]))),
            torch.tensor(np.arange(len(detections[0]["probs"])))
        )

    image_data = prepare_image_data(detections, annotations, dataset, model, mapping, device)

    results = []

    # Pre-split once (important!)
    chunks = list(chunked_image_data(image_data, chunk_size))

    for area_range_name, area_range in tqdm(area_ranges.items(), desc="Area Ranges"):
        for iou_thresh in tqdm(iou_thresholds, desc="IoU Thresholds"):

            futures = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for chunk in chunks:
                    futures.append(
                        executor.submit(
                            process_chunk,
                            chunk,
                            mapping,
                            dataset,
                            model,
                            area_range,
                            iou_thresh,
                            device,
                            scores_to_sort,
                            similarity_measures
                        )
                    )

            # ---- merge results ----
            all_scores = {k: [] for k in scores_to_sort}
            all_sims = {k: [] for k in similarity_measures}
            all_metrics = {k: [] for k in metrics}
            matched_metrics = {k: [] for k in metrics}
            all_tp = []
            total_gt = 0

            for f in futures:
                chunk_scores, chunk_sims, chunk_tp, chunk_gt, chunk_metrics, chunk_metrics_matched = f.result()

                for k in scores_to_sort:
                    all_scores[k].extend(chunk_scores[k])
                for k in similarity_measures:
                    all_sims[k].extend(chunk_sims[k])

                all_tp.extend(chunk_tp)
                total_gt += chunk_gt

                #  metrics aggregation
                for k in metrics:
                    all_metrics[k].append(chunk_metrics[k])
                    matched_metrics[k].append(chunk_metrics_matched[k])

            # concatenate over chunks
            for k in all_scores:
                all_scores[k] = torch.cat(all_scores[k])
            for k in all_sims:
                all_sims[k] = torch.cat(all_sims[k])

            bbox_tps = torch.cat(all_tp)

            for k in metrics:
                all_metrics[k] = torch.cat(all_metrics[k])
                matched_metrics[k] = torch.cat(matched_metrics[k])

            # ---- compute AUSRC ----
            for sim in similarity_measures:
                for score in scores_to_sort:

                    ausrc = compute_AUSRC(
                        all_scores[score],
                        all_sims[sim],
                        bbox_tps,
                        total_gt=total_gt
                    )
                    # compute mean metrics
                    mean_metrics_all = {f"mean_{k}_all": all_metrics[k].mean().item() for k in all_metrics}
                    mean_metrics_matched = {f"mean_{k}_matched": matched_metrics[k].mean().item() for k in matched_metrics}

                    results.append({
                        "object_size": area_range_name,
                        "iou": float(iou_thresh),
                        "score_name": score,
                        "similarity_measure": sim,
                        "AUSRC": ausrc,
                        **mean_metrics_all,
                        **mean_metrics_matched
                    })

    return results

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

    parser.add_argument("--AUSRC_result_csv", type=str, default="AUSRC_results.csv", help="CSV file to store AUSRC evaluation results")

    args = parser.parse_args()

    print(f"Evaluating model {args.model} on dataset {args.dataset} with GT {args.gt_name}")
    # Gather predictions and annotations

    detections = gather_predictions(args.dataset, args.model)
    annotations = gather_annotations(args.dataset, args.gt_file)

    # Compute soft label based metrics including AUSRC and distance measures, stores results in a CSV file for later analysis.

    area_ranges = {
        "all": (0**2, 1e5**2),
        "small": (0**2, 32**2),
        "medium": (32**2, 96**2),
        "large": (96**2, 1e5**2),
    }

    results = AUSRC_evaluation(
        args.dataset,
        args.model,
        detections,
        annotations,
        iou_thresholds= np.linspace(0.5, 0.95, 10),
        similarity_measures = ["cosine", "tv", "js"],
        scores_to_sort = ["maxprob", "entropy"],
        device="cuda",
        area_ranges=area_ranges,
        chunk_size=128,
        num_workers=32
        )
    
    df = pd.DataFrame(results)

    df.to_csv(f"results/{args.gt_name}_{args.model}_{args.AUSRC_result_csv}")