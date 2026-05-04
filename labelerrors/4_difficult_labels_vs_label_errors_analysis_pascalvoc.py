import os
import json
from glob import glob


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def is_difficult(obj):
    return str(obj.get("difficult", "0")).strip() == "1"


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_original_only_error_to_gt(error, gt_objects, iou_thr=0.5):

    if error.get("type") != "original_only":
        return None

    original_box = error.get("original_box", {})
    err_label = original_box.get("label")
    err_bbox = original_box.get("bbox")

    if err_label is None or err_bbox is None:
        return None

    best_idx = None
    best_iou = -1.0

    for idx, gt_obj in enumerate(gt_objects):
        if gt_obj["label"] != err_label:
            continue

        iou = bbox_iou(err_bbox, gt_obj["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_iou >= iou_thr:
        return best_idx

    return None


def analyze_original_only_vs_difficult(gt_dir, legt_dir, split_name="split", iou_thr=0.5):
    total_difficult = 0
    total_non_difficult = 0

    difficult_with_original_only = set()
    non_difficult_with_original_only = set()

    gt_files = sorted(glob(os.path.join(gt_dir, "*.json")))

    for gt_file in gt_files:
        img_name = os.path.basename(gt_file).replace(".json", "")
        legt_file = os.path.join(legt_dir, img_name + ".json")

        gt_data = load_json(gt_file)
        gt_objects = gt_data["objects"]

        for obj in gt_objects:
            if is_difficult(obj):
                total_difficult += 1
            else:
                total_non_difficult += 1

        if not os.path.exists(legt_file):
            continue

        legt_data = load_json(legt_file)

        for error in legt_data.get("errors", []):
            if error.get("type") != "original_only":
                continue

            matched_idx = match_original_only_error_to_gt(error, gt_objects, iou_thr=iou_thr)
            if matched_idx is None:
                continue

            key = (img_name, matched_idx)  

            if is_difficult(gt_objects[matched_idx]):
                difficult_with_original_only.add(key)
            else:
                non_difficult_with_original_only.add(key)

    diff_count = len(difficult_with_original_only)
    non_diff_count = len(non_difficult_with_original_only)

    diff_rate = 100 * diff_count / total_difficult if total_difficult > 0 else 0.0
    non_diff_rate = 100 * non_diff_count / total_non_difficult if total_non_difficult > 0 else 0.0
    ratio = diff_rate / non_diff_rate if non_diff_rate > 0 else None

    print(f"\n=== {split_name}: original_only vs difficult ===")
    print(f"Difficult objects: {diff_count}/{total_difficult} = {diff_rate:.2f}%")
    print(f"Non-difficult objects: {non_diff_count}/{total_non_difficult} = {non_diff_rate:.2f}%")
    if ratio is not None:
        print(f"Rate ratio: {ratio:.2f}x")
    else:
        print("Rate ratio: undefined")

    return {
        "split_name": split_name,
        "diff_count": diff_count,
        "total_difficult": total_difficult,
        "non_diff_count": non_diff_count,
        "total_non_difficult": total_non_difficult,
        "diff_rate": diff_rate,
        "non_diff_rate": non_diff_rate,
        "ratio": ratio,
    }


def combine_results(train_res, val_res):
    diff_count = train_res["diff_count"] + val_res["diff_count"]
    total_difficult = train_res["total_difficult"] + val_res["total_difficult"]

    non_diff_count = train_res["non_diff_count"] + val_res["non_diff_count"]
    total_non_difficult = train_res["total_non_difficult"] + val_res["total_non_difficult"]

    diff_rate = 100 * diff_count / total_difficult if total_difficult > 0 else 0.0
    non_diff_rate = 100 * non_diff_count / total_non_difficult if total_non_difficult > 0 else 0.0
    ratio = diff_rate / non_diff_rate if non_diff_rate > 0 else None

    print("\n=== combined (train + val): original_only vs difficult ===")
    print(f"Difficult objects: {diff_count}/{total_difficult} = {diff_rate:.2f}%")
    print(f"Non-difficult objects: {non_diff_count}/{total_non_difficult} = {non_diff_rate:.2f}%")
    if ratio is not None:
        print(f"Rate ratio: {ratio:.2f}x")
    else:
        print("Rate ratio: undefined")

    return {
        "split_name": "combined",
        "diff_count": diff_count,
        "total_difficult": total_difficult,
        "non_diff_count": non_diff_count,
        "total_non_difficult": total_non_difficult,
        "diff_rate": diff_rate,
        "non_diff_rate": non_diff_rate,
        "ratio": ratio,
    }


def print_paper_sentence(res):
    if res["ratio"] is None:
        print(
            f"\nPaper sentence ({res['split_name']}): "
            f"Objects marked as difficult account for all original-only errors, while "
            f"non-difficult objects show an original-only error rate of 0%."
        )
        return

    print(
        f"\nPaper sentence ({res['split_name']}): "
        f"Objects marked as difficult are substantially more likely to correspond to "
        f"original-only errors than non-difficult objects "
        f"({res['diff_rate']:.2f}% vs. {res['non_diff_rate']:.2f}%; "
        f"{res['ratio']:.2f}x)."
    )


if __name__ == "__main__":
    GT_TRAIN_DIR = "/path/to/benchmark_datasets_gt/PascalVOC/train/json"
    LEGT_TRAIN_DIR = "/path/to/benchmark_datasets_legt/PascalVOC/train/json"

    GT_VAL_DIR = "/path/to/benchmark_datasets_gt/PascalVOC/val/json"
    LEGT_VAL_DIR = "/path/to/benchmark_datasets_legt/PascalVOC/val/json"

    IOU_THR = 0.5

 
    train_res = analyze_original_only_vs_difficult(
        gt_dir=GT_TRAIN_DIR,
        legt_dir=LEGT_TRAIN_DIR,
        split_name="train",
        iou_thr=IOU_THR,
    )

    val_res = analyze_original_only_vs_difficult(
        gt_dir=GT_VAL_DIR,
        legt_dir=LEGT_VAL_DIR,
        split_name="val",
        iou_thr=IOU_THR,
    )

 
    combined_res = combine_results(train_res, val_res)

    print_paper_sentence(train_res)
    print_paper_sentence(val_res)
    print_paper_sentence(combined_res)




