
import json
from glob import glob

from helpers_legt import *


#############################################################################
DATASET = "Kitti"
CLASS_NAMES = ["car", "van", "truck", "pedestrian", "personsitting", "cyclist", "tram"]
MODES = ["train", "val"]
GT = f"/path/to/benchmark_datasets_gt/{DATASET}"
VGT = f"/path/to/benchmark_datasets_vgt/{DATASET}"
ALL_VGT = f"/path/to/benchmark_datasets_vgt_all_bboxes/{DATASET}"
LEGT = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}"
#############################################################################




def compute_ioa_with_dont_care():
    for mode in MODES:
        legt_files = sorted(glob(f"{LEGT}/{mode}/json/*.json"))
        dont_care_path = f"{GT}/{mode}/dontcare"
        
        for legt_file in legt_files:
            legt_dicti = json.load(open(legt_file))
            errors = legt_dicti["errors"]
            
            img_name = legt_file.split("/")[-1].replace(".json", "")
                
            dontcare_objs = json.load(open(f"{dont_care_path}/{img_name}.json"))["objects"]
            dontcare_bboxes = [obj["bbox"] for obj in dontcare_objs]
        
            for error in errors:        
                if error["type"] == "original_only":
                    box_dict = error["original_box"]
                else:
                    box_dict = error["validated_box"]

                box = box_dict["bbox"]

                max_ioa = 0.0
                for dc_box in dontcare_bboxes:
                    ioa = compute_ioa(box, dc_box)
                    if ioa > max_ioa:
                        max_ioa = ioa

                box_dict["ioa_with_dont_care"] = max_ioa
                
            legt_dicti["errors"] = errors

            with open(legt_file, "w") as f:
                json.dump(legt_dicti, f, indent=4)


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
    compute_ioa_with_dont_care()