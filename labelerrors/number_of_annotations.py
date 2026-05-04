from collections import Counter
import pandas as pd
import json


def count_annotations_per_box(obj):
    return sum(obj["frequencies"].values())


def analyze_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    dataset_name = data["dataset_name"]
    objects = data["objects"]

    counts = []

    for image_id, objs in objects.items():
        for obj in objs:
            counts.append(count_annotations_per_box(obj))

    total_boxes = len(counts)
    counter = Counter(counts)

    print(dataset_name, counter)

    def pct(n):
        return 100 * n / total_boxes if total_boxes > 0 else 0

    n_11 = counter.get(11, 0)
    n_22 = counter.get(22, 0)
    n_33 = counter.get(33, 0)
    n_44 = counter.get(44, 0)

    return {
        "Dataset": dataset_name,
        "#Boxes": total_boxes,

        "#11": n_11,
        "%11": pct(n_11),

        "#22": n_22,
        "%22": pct(n_22),

        "#33": n_33,
        "%33": pct(n_33),

        "#44": n_44,
        "%44": pct(n_44),
    }


def analyze_all(dataset_files):
    results = []

    for dataset_name, path in dataset_files.items():
        results.append(analyze_dataset(path))

    df = pd.DataFrame(results)

    percentage_cols = ["%11", "%22", "%33", "%44"]
    df[percentage_cols] = df[percentage_cols].round(2)

    return df



dataset_files = {
    "Cityscapes": "/path/to/benchmark_paper_post_processing/our_datasets/soft_Cityscapes_train.json",
    #"Cityscapes_val": "/path/to/benchmark_paper_post_processing/our_datasets/soft_Cityscapes_val.json",
    "KITTI": "/path/to/benchmark_paper_post_processing/our_datasets/soft_Kitti_2D_train_val.json",
    "PascalVOC": "/path/to/benchmark_paper_post_processing/our_datasets/soft_PascalVOC_2012_detection_trainval.json",
    "COCO": "/path/to/benchmark_paper_post_processing/our_datasets/soft_COCO_2017_val.json",
}

df = analyze_all(dataset_files)
print(df)
#print(df.to_latex(index=False))
