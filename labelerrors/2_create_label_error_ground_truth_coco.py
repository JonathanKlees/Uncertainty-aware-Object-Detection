

import json
from glob import glob

from helpers_legt import *


#############################################################################
DATASET = "COCO"
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
MODES = ["val"]
GT = f"/path/to/benchmark_datasets_gt/{DATASET}"
VGT = f"/path/to/benchmark_datasets_vgt/{DATASET}"
ALL_VGT = f"/path/to/benchmark_datasets_vgt_all_bboxes/{DATASET}"
LEGT = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}"
#############################################################################





def filter_original_only():
    for mode in MODES:
        gt_path = f"{GT}/{mode}/json"
        all_vgt_path = f"{ALL_VGT}/{mode}/json"
        legt_path = f"{LEGT}/{mode}/json"
        
        all_vgt_files = sorted(glob(f"{all_vgt_path}/*.json"))
  
        for vgt_file in all_vgt_files:
            img_name = vgt_file.split("/")[-1].replace(".json", "")
            
            original = json.load(open(f"{gt_path}/{img_name}.json"))
            validated = json.load(open(vgt_file))
            legt = json.load(open(f"{legt_path}/{img_name}.json"))

            legt = extract_original_only(original, validated, legt, CLASS_NAMES)

            with open(f"{legt_path}/{img_name}.json", "w") as f:
                json.dump(legt, f, indent=4)
                

def create_legt():
    for mode in MODES:
        gt_path = f"{GT}/{mode}/json"
        vgt_path = f"{VGT}/{mode}/json"
        legt_path = f"{LEGT}/{mode}/json"
        
        vgt_files = sorted(glob(f"{vgt_path}/*.json"))
  
        for vgt_file in vgt_files:
            img_name = vgt_file.split("/")[-1].replace(".json", "")
            
            original = json.load(open(f"{gt_path}/{img_name}.json"))
            validated = json.load(open(vgt_file))

            legt = extract_label_errors(original, validated, CLASS_NAMES)

            with open(f"{legt_path}/{img_name}.json", "w") as f:
                json.dump(legt, f, indent=4)
                


if __name__ == "__main__":
    create_legt()
    filter_original_only()
    save_bbox_heights(MODES, LEGT)