import os
import json
from pathlib import Path

DATASET = "Cityscapes"
MODE = "train"
INPUT_DIR = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}/{MODE}/json"
OUTPUT_DIR_V1 = f"/path/to/benchmark_datasets_legt_second_round_variant1/{DATASET}/{MODE}/json"
OUTPUT_DIR_V2 = f"/path/to/benchmark_datasets_legt_second_round_variant2/{DATASET}/{MODE}/json"


def get_relevant_box(error):
    err_type = error.get("type")

    if err_type in ["missing", "misaligned", "classification"]:
        return error.get("validated_box", {})
    elif err_type == "original_only":
        return error.get("original_box", {})
    else:
        return {}

def filter_errors(errors, min_prob=None, min_height=None):
    filtered = []

    for err in errors:
        box = get_relevant_box(err)

        prob = box.get("prob", None)
        bbox_height = box.get("bbox_height", None)

        if min_prob is not None:
            if prob is None or prob < min_prob:
                continue

        if min_height is not None:
            if bbox_height is None or bbox_height < min_height:
                continue

        filtered.append(err)

    return filtered

def process_directory(input_dir, output_dir, min_prob=None, min_height=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files.")

    total_before = 0
    total_after = 0

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        errors = data.get("errors", [])
        total_before += len(errors)

        filtered_errors = filter_errors(
            errors,
            min_prob=min_prob,
            min_height=min_height
        )
        total_after += len(filtered_errors)

        new_data = data.copy()
        new_data["errors"] = filtered_errors

        rel_path = json_file.relative_to(input_dir)
        out_file = output_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)

    print(f"Saved filtered files to: {output_dir}")
    print(f"Errors before: {total_before}")
    print(f"Errors after : {total_after}")


if __name__ == "__main__":
    
    process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR_V1,
        min_prob=0.5,
        min_height=None
    )

    process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR_V2,
        min_prob=0.8,
        min_height=40
    )