import json
from glob import glob



def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def compute_ioa(box, dontcare_box):
    inter = intersection(box, dontcare_box)
    a = area(box)
    if a == 0:
        return 0.0
    return inter / a


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    if union == 0:
        return 0.0

    return inter_area / union

def greedy_match(original_objs, validated_objs, iou_threshold=0.5):
    matches = []
    used_o = set()
    used_v = set()
    pairs = []

    for i, o in enumerate(original_objs):
        for j, v in enumerate(validated_objs):
            iou = compute_iou(o["bbox"], v["bbox"])
            pairs.append((iou, i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    for iou, i, j in pairs:
        if i in used_o or j in used_v:
            continue
        if iou < iou_threshold:
            continue

        matches.append((i, j, iou))
        used_o.add(i)
        used_v.add(j)

    return matches

def count_label_errors(modes, legt_base_path):
    for mode in modes:
        legt_path = f"{legt_base_path}/{mode}/json"
        
        legt_files = sorted(glob(f"{legt_path}/*.json"))
        
        miss_count = 0
        misaligned_count = 0
        class_count = 0
        original_only_count = 0
        
        for legt_file in legt_files:
            legt = json.load(open(legt_file))
            errors = legt["errors"]
            
            for error in errors:
                type = error["type"]
                
                if type == "missing":
                    miss_count = miss_count + 1
                elif type == "misaligned":
                    misaligned_count = misaligned_count + 1
                elif type == "classification":
                    class_count = class_count + 1
                elif type == "original_only":
                    original_only_count = original_only_count + 1
                    
        print(f"\nMode: {mode}")
        print(f"Missing: {miss_count}")
        print(f"Misaligned: {misaligned_count}")
        print(f"Classification: {class_count}")
        print(f"Original only: {original_only_count}")
        

def save_bbox_heights(modes, legt_path):
    for mode in modes:
        legt_files = sorted(glob(f"{legt_path}/{mode}/json/*.json"))
        
        for legt_file in legt_files:
            legt_dicti = json.load(open(legt_file))
            errors = legt_dicti["errors"]
            
            for err in errors:
                if err["type"] != "original_only":
                    vgt_box = err["validated_box"]["bbox"]
                    x_min, y_min, x_max, y_max = vgt_box
                    height = y_max - y_min
                
                    err["validated_box"]["bbox_height"] = height
                else:
                    org_box = err["original_box"]["bbox"]
                    x_min, y_min, x_max, y_max = org_box
                    height = y_max - y_min
                    
                    err["original_box"]["bbox_height"] = height
                    
            legt_dicti["errors"] = errors
            
            with open(legt_file, "w") as f:
                json.dump(legt_dicti, f, indent=4)
                

def extract_label_errors(original_json, validated_json, class_names):
    original_objs = original_json["objects"]
    rel_original_objs = [obj for obj in original_objs if obj["label"] in class_names]
    
    validated_objs = validated_json["objects"]

    matches = greedy_match(rel_original_objs, validated_objs, iou_threshold=0.5)

    matched_v = set([m[1] for m in matches])

    max_iou_v = []
    for v in validated_objs:
        if len(rel_original_objs) == 0:
            max_iou_v.append(0)
        else:
            max_iou_v.append(
                max(compute_iou(v["bbox"], o["bbox"]) for o in rel_original_objs)
            )

    errors = []

    """
        1. Missing + Misaligned (validated perspective)
    """
    
    for j, v in enumerate(validated_objs):
        if j in matched_v:
            continue

        iou = max_iou_v[j]

        if iou <= 0.1:
            errors.append({
                "type": "missing",
                "validated_box": v,
                "max_iou": iou
            })
        elif 0.1 < iou < 0.5:
            errors.append({
                "type": "misaligned",
                "validated_box": v,
                "max_iou": iou
            })

    """
        2. Classification Error
    """
    
    for i, j, iou in matches:
        o = rel_original_objs[i]
        v = validated_objs[j]

        if iou >= 0.7 and o["label"] != v["label"]:
            errors.append({
                "type": "classification",
                "original_box": o,
                "validated_box": v,
                "iou": iou
            })

    return {
        "height": original_json.get("height", original_json.get("imgHeight")),
        "width": original_json.get("width", original_json.get("imgWidth")),
        "errors": errors
    }


def extract_original_only(original_json, validated_json, legt_json, class_names):
    original_objs = original_json["objects"]
    rel_original_objs = [obj for obj in original_objs if obj["label"] in class_names]
    
    errors = legt_json["errors"]
    validated_objs = validated_json["objects"]

    max_iou_o = []
    for o in rel_original_objs:
        if len(validated_objs) == 0:
            max_iou_o.append(0)
        else:
            max_iou_o.append(
                max(compute_iou(o["bbox"], v["bbox"]) for v in validated_objs)
           )
    
    """
        3. Original only
    """
    
    for i, o in enumerate(rel_original_objs):
        if max_iou_o[i] <= 0.1:
            errors.append({
                "type": "original_only",
                "original_box": o,
                "max_iou": max_iou_o[i]
            })

    legt_json["errors"] = errors
    
    return legt_json