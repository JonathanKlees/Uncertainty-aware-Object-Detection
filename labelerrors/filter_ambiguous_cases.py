import json
import numpy as np
from collections import defaultdict


def has_no_unique_winner(soft_label, tol=1e-6):
    arr = np.array(soft_label, dtype=float)
    max_val = arr.max()
    winner_indices = np.where(np.isclose(arr, max_val, atol=tol))[0]
    return len(winner_indices) > 1, winner_indices.tolist(), float(max_val)


def collect_ambiguous_cases(dataset_files, output_path="ambiguous_tie_cases.json"):

    ambiguous_cases = defaultdict(list)

    for dataset_name, path in dataset_files.items():
        with open(path, "r") as f:
            data = json.load(f)

        output_dataset_name = "Cityscapes" if dataset_name.startswith("Cityscapes") else dataset_name

        objects = data["objects"]

        for image_name, objs in objects.items():
            for obj in objs:
                soft_label = obj["postprocessed_soft_label"]

                is_tie, winner_indices, max_val = has_no_unique_winner(soft_label)

                if is_tie:

                    ambiguous_cases[output_dataset_name].append({
                        "image_name": image_name,
                        "bbox": obj.get("bbox"),
                        "mo_id": obj.get("mo_id"),
                        "postprocessed_soft_label": soft_label,
                        #"tied_classes": winner_classes,
                        "tied_indices": winner_indices,
                        "tie_probability": max_val,
                        "proposed_class": obj.get("proposed_class")
                    })

    ambiguous_cases = dict(ambiguous_cases)

    with open(output_path, "w") as f:
        json.dump(ambiguous_cases, f, indent=2)

    return ambiguous_cases


dataset_files = {
    "Cityscapes": "soft_Cityscapes_train.json",
    "Cityscapes_val": "soft_Cityscapes_val.json",
    "KITTI": "soft_Kitti_2D_train_val.json",
    "PascalVOC": "soft_PascalVOC_2012_detection_trainval.json",
    "COCO": "soft_COCO_2017_val.json",
}






if __name__ == '__main__':
    dataset_files = {
        "Cityscapes": "/path/to/benchmark_paper_post_processing/our_datasets/soft_Cityscapes_train.json",
        "Cityscapes_val": "/path/to/benchmark_paper_post_processing/our_datasets/soft_Cityscapes_val.json",
        "KITTI": "/path/to/benchmark_paper_post_processing/our_datasets/soft_Kitti_2D_train_val.json",
        "PascalVOC": "/path/to/benchmark_paper_post_processing/our_datasets/soft_PascalVOC_2012_detection_trainval.json",
        "COCO": "/path/to/benchmark_paper_post_processing/our_datasets/soft_COCO_2017_val.json",
    }
    
    ambiguous_cases = collect_ambiguous_cases(
        dataset_files,
        output_path="ambiguous_cases.json"
    )

    for dataset_name, cases in ambiguous_cases.items():
        print(f"{dataset_name}: {len(cases)} ambiguous tie cases")