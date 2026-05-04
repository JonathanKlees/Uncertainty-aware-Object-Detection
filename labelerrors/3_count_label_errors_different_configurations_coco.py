import os
import json
from glob import glob
from itertools import product

import pandas as pd

DATASET = "COCO"
VAL_DIR = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}/val/json"
OUT_DIR = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}/labelerrorcounts"

OUTPUT_CSV = f"{DATASET}_label_error_counts.csv"
OUTPUT_LATEX_VAL = f"{DATASET}_label_error_table_val.tex"


def get_relevant_box(error):
    """
    - missing        -> validated_box
    - misaligned     -> validated_box
    - classification -> validated_box
    - original_only  -> original_box
    """
    err_type = error["type"]

    if err_type in ["missing", "misaligned", "classification"]:
        return error.get("validated_box", {})
    elif err_type == "original_only":
        return error.get("original_box", {})
    else:
        return {}


def parse_error_file(legt_path, split_name):
    with open(legt_path, "r") as f:
        legt_file = json.load(f)

    img_name = os.path.basename(legt_path).replace(".json", "")
    rows = []

    for err in legt_file.get("errors", []):
        err_type = err.get("type")
        rel_box = get_relevant_box(err)

        bbox = rel_box.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        if err_type == "original_only":
            prob = None
        else:
            prob = rel_box.get("prob")

        bbox_height = rel_box.get("bbox_height")
        if bbox_height is None:
            x1, y1, x2, y2 = bbox
            bbox_height = y2 - y1

        rows.append({
            "split": split_name,
            "image_name": img_name,
            "error_type": err_type,
            "prob": prob,
            "bbox_height": bbox_height,
        })

    return rows


def load_all_errors(val_dir):
    rows = []

    val_files = sorted(glob(os.path.join(val_dir, "*.json")))

    for path in val_files:
        rows.extend(parse_error_file(path, "val"))

    return pd.DataFrame(rows)


PROB_FILTERS = {
    "ge_0.5": lambda df: df["prob"] >= 0.5,
    "ge_0.8": lambda df: df["prob"] >= 0.8,
}

HEIGHT_FILTERS = {
    "all": lambda df: pd.Series(True, index=df.index),
    "ge_25": lambda df: df["bbox_height"] >= 25,
    "ge_40": lambda df: df["bbox_height"] >= 40,
}

ERROR_TYPES = ["missing", "misaligned", "classification", "original_only"]
SPLITS = ["val"]
PROB_ORDER = ["ge_0.5", "ge_0.8"]
HEIGHT_ORDER = ["all", "ge_25", "ge_40"]


def count_all_combinations(df):
    results = []

    for split, err_type, h_name in product(SPLITS, ERROR_TYPES, HEIGHT_ORDER):
        sub = df[
            (df["split"] == split) &
            (df["error_type"] == err_type)
        ]

        if len(sub) == 0:
            if err_type == "original_only":
                results.append({
                    "split": split,
                    "error_type": err_type,
                    "prob_filter": "none",
                    "height_filter": h_name,
                    "count": 0,
                })
            else:
                for prob_name in PROB_ORDER:
                    results.append({
                        "split": split,
                        "error_type": err_type,
                        "prob_filter": prob_name,
                        "height_filter": h_name,
                        "count": 0,
                    })
            continue

        if err_type == "original_only":
            mask = HEIGHT_FILTERS[h_name](sub)
            count = int(mask.sum())

            results.append({
                "split": split,
                "error_type": err_type,
                "prob_filter": "none",
                "height_filter": h_name,
                "count": count,
            })

        else:
            for prob_name in PROB_ORDER:
                mask = (
                    PROB_FILTERS[prob_name](sub) &
                    HEIGHT_FILTERS[h_name](sub)
                )
                count = int(mask.sum())

                results.append({
                    "split": split,
                    "error_type": err_type,
                    "prob_filter": prob_name,
                    "height_filter": h_name,
                    "count": count,
                })

    return pd.DataFrame(results)

def pretty_height_name(height_name):
    mapping = {
        "all": "Arbitrary",
        "ge_25": "$\\geq 25$ pixels",
        "ge_40": "$\\geq 40$ pixels",
    }
    return mapping.get(height_name, height_name)


def get_count(res_df, split, err_type, prob_name, h_name):
    sub = res_df[
        (res_df["split"] == split) &
        (res_df["error_type"] == err_type) &
        (res_df["prob_filter"] == prob_name) &
        (res_df["height_filter"] == h_name)
    ]
    if len(sub) == 0:
        return 0
    return int(sub["count"].iloc[0])

def build_latex_table_split(res_df, split):
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        f"\\caption{{Number of identified label errors on the {split} split in the COCO dataset depending on the probability threshold for soft label annotations and the minimal height of considered objects.}}"
    )
    lines.append(f"\\label{{tab:{split}_error_counts}}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{lccc|ccc}")
    lines.append("\\toprule")

    lines.append(
        "Probability threshold & "
        "\\multicolumn{3}{c|}{$p \\geq 0.5$} & "
        "\\multicolumn{3}{c}{$p \\geq 0.8$} \\\\"
    )
    lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")

    lines.append(
        "Object Height & Missing & Misaligned & Classification & Missing & Misaligned & Classification \\\\"
    )
    lines.append("\\midrule")

    for h_name in HEIGHT_ORDER:
        m_05 = get_count(res_df, split, "missing", "ge_0.5", h_name)
        mis_05 = get_count(res_df, split, "misaligned", "ge_0.5", h_name)
        c_05 = get_count(res_df, split, "classification", "ge_0.5", h_name)

        m_08 = get_count(res_df, split, "missing", "ge_0.8", h_name)
        mis_08 = get_count(res_df, split, "misaligned", "ge_0.8", h_name)
        c_08 = get_count(res_df, split, "classification", "ge_0.8", h_name)

        lines.append(
            f"{pretty_height_name(h_name)} & "
            f"{m_05} & {mis_05} & {c_05} & "
            f"{m_08} & {mis_08} & {c_08} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_all_errors(VAL_DIR)

    if len(df) == 0:
        print("No errors found.")
        return

    print("\nPreview of parsed errors:")
    print(df.head(10))

    res_df = count_all_combinations(df)

    print("\nPreview of counts:")
    print(res_df.head(20))

    res_df.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)

    latex_val = build_latex_table_split(res_df, "val")
    with open(os.path.join(OUT_DIR, OUTPUT_LATEX_VAL), "w") as f:
        f.write(latex_val)


if __name__ == "__main__":
    main()