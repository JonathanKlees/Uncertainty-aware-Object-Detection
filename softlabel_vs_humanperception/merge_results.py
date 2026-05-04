import json
import os

import json


def merge_human_agreements(files):
    """
    files: List of loaded JSONs (dicts)
    """

    # Base = first file
    merged = files[0]

    # Mapping: box_id -> sample
    id_to_sample = {
        s["box_id"]: s for s in merged["all_samples"]
    }

    # Iterate through all remaining files
    for file in files[1:]:
        for sample in file["all_samples"]:
            box_id = sample["box_id"]

            if box_id not in id_to_sample:
                continue  # optional: warning

            target = id_to_sample[box_id]

            # Source & Target Agreements
            src = sample.get("human_class_agreement", {})
            tgt = target.get("human_class_agreement", {})

            # Merge (without changing other keys!)
            for reviewer, value in src.items():
                tgt[reviewer] = value

            target["human_class_agreement"] = tgt

    return merged


if __name__ == '__main__':
    dataset = "PascalVOC"
    
    file1 = json.load(open(f"review_decisions/{dataset}_class_or_not_review_reviewer_3.json"))
    file2 = json.load(open(f"review_decisions/{dataset}_class_or_not_review_reviewer_2.json"))
    file3 = json.load(open(f"review_decisions/{dataset}_class_or_not_review_reviewer_1.json"))

    merged = merge_human_agreements([file1, file2, file3])

    with open(f"review_decisions/{dataset}_class_or_not_review_merged.json", "w") as f:
        json.dump(merged, f, indent=4)