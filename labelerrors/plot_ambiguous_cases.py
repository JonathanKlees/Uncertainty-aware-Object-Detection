import json
import cv2
import os
import numpy as np


def get_image_path(image_roots, image_name):
    has_ext = os.path.splitext(image_name)[1] != ""

    for root in image_roots:
        if has_ext:
            path = os.path.join(root, image_name)
            if os.path.exists(path):
                return path
        else:
            for ext in [".png", ".jpg", ".jpeg"]:
                path = os.path.join(root, image_name + ext)
                if os.path.exists(path):
                    return path

    return None

def crop_with_context(img, bbox, pad_ratio=0.5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    cx1 = int(max(0, x1 - pad_x))
    cy1 = int(max(0, y1 - pad_y))
    cx2 = int(min(w, x2 + pad_x))
    cy2 = int(min(h, y2 + pad_y))

    cropped = img[cy1:cy2, cx1:cx2].copy()

    new_bbox = [
        x1 - cx1,
        y1 - cy1,
        x2 - cx1,
        y2 - cy1
    ]

    return cropped, new_bbox


def plot_ambiguous_cases_cropped(
    ambiguous_json_path,
    image_roots,
    output_dirs,
    classes_per_dataset
):
    with open(ambiguous_json_path, "r") as f:
        data = json.load(f)

    for dataset_name, cases in data.items():

        classes = classes_per_dataset[dataset_name]
        image_root = image_roots[dataset_name]
        output_dir = output_dirs[dataset_name]

        os.makedirs(output_dir, exist_ok=True)

        for i, case in enumerate(cases):

            image_name = case["image_name"]
            bbox = case["bbox"]
            tied_indices = case["tied_indices"]

            tied_classes = [classes[idx] for idx in tied_indices]
            label_text = " / ".join(tied_classes)

            roots = image_roots[dataset_name]  # Liste
            img_path = get_image_path(roots, image_name)

            if img_path is None:
                print(f"[{dataset_name}] Image not found: {image_name}")
                continue

            if not os.path.exists(img_path):
                print(f"[{dataset_name}] Image not found: {img_path}")
                continue

            img = cv2.imread(img_path)

        
            cropped, bbox = crop_with_context(img, bbox, pad_ratio=2.0)
            x1, y1, x2, y2 = bbox

            cv2.rectangle(cropped, (x1, y1), (x2, y2), (0, 0, 255), 2)

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            text_x = max(0, x1)
            text_y = max(th + 5, y1 - 5)

            cv2.rectangle(
                cropped,
                (text_x, text_y - th - 4),
                (text_x + tw, text_y),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                cropped,
                label_text,
                (text_x, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            out_path = os.path.join(output_dir, f"{i}.png")
            cv2.imwrite(out_path, cropped)

        print(f"{dataset_name}: saved {len(cases)} images")
        
if __name__ == '__main__':
    
    image_roots = {
        "Cityscapes": [
            "/path/to/benchmark_datasets_gt/Cityscapes/train/rgb",
            "/path/to/benchmark_datasets_gt/Cityscapes/val/rgb"
        ],
        "KITTI": [
            "/path/to/benchmark_datasets_gt/Kitti/train/rgb",
            "/path/to/benchmark_datasets_gt/Kitti/val/rgb"
        ],
        "PascalVOC": [
            "/path/to/benchmark_datasets_gt/PascalVOC/train/rgb",
            "/path/to/benchmark_datasets_gt/PascalVOC/val/rgb"
        ],
        "COCO": [
            "/path/to/datasets/COCO/2017/val2017"
        ],
    }

    output_dirs = {
        "Cityscapes": "/path/to/benchmark_datasets_images/ambiguous_cases/Cityscapes",
        "KITTI": "/path/to/benchmark_datasets_images/ambiguous_cases/Kitti",
        "PascalVOC": "/path/to/benchmark_datasets_images/ambiguous_cases/PascalVOC",
        "COCO": "/path/to/benchmark_datasets_images/ambiguous_cases/COCO",
    }

    classes_per_dataset = {
        "Cityscapes": ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "cantsolve"],
        "KITTI": ["car", "van", "truck", "pedestrian", "personsitting", "cyclist", "tram", "cantsolve"],
        "PascalVOC": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', "cantsolve"],
        "COCO": ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'trafficlight', 'firehydrant',
               'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat',
               'baseballglove', 'skateboard', 'surfboard', 'tennisracket',
               'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cellphone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush', "cantsolve"]
    }


    plot_ambiguous_cases_cropped(
        ambiguous_json_path="ambiguous_cases.json",
        image_roots=image_roots,
        output_dirs=output_dirs,
        classes_per_dataset=classes_per_dataset
    )
