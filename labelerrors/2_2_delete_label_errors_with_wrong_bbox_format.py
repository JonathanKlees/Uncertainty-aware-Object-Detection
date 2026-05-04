import json
from glob import glob


def is_valid_xyxy_bbox(bbox, width, height):
    if bbox is None or len(bbox) != 4:
        return False

    x_min, y_min, x_max, y_max = bbox

    if not all(isinstance(v, (int, float)) for v in bbox):
        return False

    if x_max <= x_min or y_max <= y_min:
        return False


    if x_min < 0 or y_min < 0:
        return False
    if x_max > width or y_max > height:
        return False

    return True


def get_error_bbox(error):
    if error["type"] == "original_only":
        return error.get("original_box", {}).get("bbox")
    return error.get("validated_box", {}).get("bbox")


def clean_json_folder(path):
    json_files = sorted(glob(f"{path}/*.json"))

    total_removed = 0

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        width = data["width"]
        height = data["height"]
        errors = data.get("errors", [])

        cleaned_errors = []
        removed_here = 0

        for err in errors:
            bbox = get_error_bbox(err)

            if is_valid_xyxy_bbox(bbox, width, height):
                cleaned_errors.append(err)
            else:
                removed_here += 1

        if removed_here > 0:
            print(f"{file} -> removed {removed_here} invalid boxes")

        total_removed += removed_here
        data["errors"] = cleaned_errors

        with open(file, "w") as f:
            json.dump(data, f, indent=4)

    print(f"\nTotal removed invalid boxes: {total_removed}")


if __name__ == '__main__':
    dataset = "COCO"
    modes = ["val"]
    
    for mode in modes:    
        clean_json_folder(f"/path/to/benchmark_datasets_legt_second_round/{dataset}/{mode}/json")