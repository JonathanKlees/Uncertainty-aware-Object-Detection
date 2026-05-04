import os
import json
from glob import glob
from itertools import product

import pandas as pd

DATASET = "Kitti"
BASE_PATH = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}"

TRAIN_DIR = f"{BASE_PATH}/train/json"
VAL_DIR = f"{BASE_PATH}/val/json"
OUT_DIR = f"{BASE_PATH}/labelerrorcounts"

OUTPUT_CSV = f"{DATASET}_label_error_counts.csv"
OUTPUT_LATEX_TRAIN = f"{DATASET}_label_error_table_train.tex"
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
        return error["validated_box"]
    elif err_type == "original_only":
        return error["original_box"]
    else:
        return {}


def bbox_height_from_bbox(bbox):
    return bbox[3] - bbox[1]


def parse_error_file(legt_path, split_name):
    with open(legt_path, "r") as f:
        legt_file = json.load(f)

    img_name = os.path.basename(legt_path).replace(".json", "")
    rows = []

    for err in legt_file["errors"]:
        err_type = err["type"]
        rel_box = get_relevant_box(err)

        bbox = rel_box.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        if err_type == "original_only":
            prob = None
        else:
            prob = rel_box.get("prob", None)

        row = {
            "split": split_name,
            "image_name": img_name,
            "error_type": err_type,
            "prob": prob,
            "bbox_height": rel_box.get("bbox_height", bbox_height_from_bbox(bbox)),
            "dontcare_ioa": rel_box.get("ioa_with_dont_care", 0.0),
            "label": rel_box.get("label", None),
        }
        rows.append(row)

    return rows


def load_all_errors(train_dir, val_dir):
    rows = []

    train_files = sorted(glob(os.path.join(train_dir, "*.json")))
    val_files = sorted(glob(os.path.join(val_dir, "*.json")))

    for path in train_files:
        rows.extend(parse_error_file(path, "train"))

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

DONTCARE_FILTERS = {
    "all": lambda df: pd.Series(True, index=df.index),
    "le_05": lambda df: df["dontcare_ioa"] <= 0.5,
}



ERROR_TYPES = ["missing", "misaligned", "classification", "original_only"]
TABLE_ERROR_TYPES = ["missing", "misaligned", "classification"]

SPLITS = ["train", "val"]
PROB_ORDER = ["ge_0.5", "ge_0.8"]
HEIGHT_ORDER = ["all", "ge_25", "ge_40"]
DONTCARE_ORDER = ["all", "le_05"]



def count_all_combinations(df):
    results = []

    for split, err_type, prob_name, h_name, dc_name in product(
        SPLITS, ERROR_TYPES, PROB_ORDER, HEIGHT_ORDER, DONTCARE_ORDER
    ):
        sub = df[
            (df["split"] == split) &
            (df["error_type"] == err_type)
        ]

        if len(sub) > 0:
            if err_type == "original_only":
            
                mask = (
                    HEIGHT_FILTERS[h_name](sub) &
                    DONTCARE_FILTERS[dc_name](sub)
                )
            else:
                mask = (
                    PROB_FILTERS[prob_name](sub) &
                    HEIGHT_FILTERS[h_name](sub) &
                    DONTCARE_FILTERS[dc_name](sub)
                )

            count = int(mask.sum())
        else:
            count = 0

        results.append({
            "split": split,
            "error_type": err_type,
            "prob_filter": prob_name,
            "height_filter": h_name,
            "dontcare_filter": dc_name,
            "count": count,
        })

    return pd.DataFrame(results)


def get_count(res_df, split, err_type, prob_name, h_name, dc_name):
    sub = res_df[
        (res_df["split"] == split) &
        (res_df["error_type"] == err_type) &
        (res_df["prob_filter"] == prob_name) &
        (res_df["height_filter"] == h_name) &
        (res_df["dontcare_filter"] == dc_name)
    ]
    if len(sub) == 0:
        return 0
    return int(sub["count"].iloc[0])


def build_latex_table_missing_misaligned_classification(res_df, split):
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        f"\\caption{{Number of identified label errors on the {split} split in the KITTI dataset depending on the probability threshold for soft label annotations and the minimal height of considered objects.}}"
    )
    lines.append(f"\\label{{tab:{split}_missing_misaligned_classification}}")
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

    # considering all objects
    lines.append("\\multicolumn{7}{c}{\\textit{Considering all objects}} \\\\")
    for h_name, h_label in [
        ("all", "Arbitrary"),
        ("ge_25", "$\\geq 25$ pixels"),
        ("ge_40", "$\\geq 40$ pixels"),
    ]:
        vals = []
        for prob_name in PROB_ORDER:
            for err_type in TABLE_ERROR_TYPES:
                vals.append(
                    str(
                        get_count(
                            res_df=res_df,
                            split=split,
                            err_type=err_type,
                            prob_name=prob_name,
                            h_name=h_name,
                            dc_name="all",
                        )
                    )
                )

        lines.append(f"{h_label} & " + " & ".join(vals) + " \\\\")

    lines.append("\\midrule")

    # considering only objects outside don't care
    lines.append("\\multicolumn{7}{c}{\\textit{Considering only objects outside of 'Don't Care' regions}} \\\\")
    for h_name, h_label in [
        ("all", "Arbitrary"),
        ("ge_25", "$\\geq 25$ pixels"),
        ("ge_40", "$\\geq 40$ pixels"),
    ]:
        vals = []
        for prob_name in PROB_ORDER:
            for err_type in TABLE_ERROR_TYPES:
                vals.append(
                    str(
                        get_count(
                            res_df=res_df,
                            split=split,
                            err_type=err_type,
                            prob_name=prob_name,
                            h_name=h_name,
                            dc_name="le_05",
                        )
                    )
                )

        lines.append(f"{h_label} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_all_errors(TRAIN_DIR, VAL_DIR)

    if len(df) == 0:
        print("No errors found.")
        return

    res_df = count_all_combinations(df)

    res_df.to_csv(os.path.join(OUT_DIR, OUTPUT_CSV), index=False)

    latex_train = build_latex_table_missing_misaligned_classification(res_df, "train")
    with open(os.path.join(OUT_DIR, OUTPUT_LATEX_TRAIN), "w") as f:
        f.write(latex_train)

    latex_val = build_latex_table_missing_misaligned_classification(res_df, "val")
    with open(os.path.join(OUT_DIR, OUTPUT_LATEX_VAL), "w") as f:
        f.write(latex_val)


if __name__ == "__main__":
    main()