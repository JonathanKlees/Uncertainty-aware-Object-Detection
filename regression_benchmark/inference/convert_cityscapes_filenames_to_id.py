# map cityscapes filenames to ids in predictions folder 
# This is a quick fix to convert the predicted filenames (which are based on the original image filenames) to the corresponding image ids used in the COCO annotations, so that we can evaluate using COCO metrics.
import json
import os

with open("/path/to/datasets/Cityscapes/val/instances_val.json", "r") as f:
    data = json.load(f)

filename_to_id = {img["file_name"].split(".")[0] : img["id"] for img in data["images"]}

for model in ["frcnn", "owlvit", "yolov8l", "rtdetr-l", "yolov8l-worldv2"] + ["yolov8l_hard", "yolov8l_soft", "rtdetr-l_hard", "rtdetr-l_soft"]:

    pred_dir = f"predictions/{model}/cityscapes/"

    for filename in os.listdir(pred_dir):
        if not filename.endswith(".json"):
            continue

        name, ext = os.path.splitext(filename)

        try:
            new_name = str(filename_to_id[name]) + ext  # remove leading zeros
        except KeyError:
            print(f"Skipping {filename} (filename not found in mapping)")
            continue

        old_path = os.path.join(pred_dir, filename)
        new_path = os.path.join(pred_dir, new_name)

        os.rename(old_path, new_path)