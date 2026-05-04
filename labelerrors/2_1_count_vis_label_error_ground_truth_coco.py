from helpers_vis import *
from helpers_legt import *

#########################################################################
DATASET = "COCO"
MODES = ["val"]
IMG_SUFFIX = ".jpg"
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
ERROR_TYPES = ["missing", "misaligned", "classification", "original_only"]
RGB = f"/path/to/datasets/COCO/2017/val2017"
GT = f"/path/to/benchmark_datasets_gt/{DATASET}"
VGT = f"/path/to/benchmark_datasets_vgt/{DATASET}"
ALL_VGT = f"/path/to/benchmark_datasets_vgt_all_bboxes/{DATASET}"
LEGT = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}"
#########################################################################


def vis_label_errors():
    vis_main_errors(MODES, RGB, IMG_SUFFIX, GT, VGT, LEGT, CLASS_NAMES)
    vis_original_only(MODES, RGB, IMG_SUFFIX, GT, ALL_VGT, LEGT, CLASS_NAMES)
    vis_random_original_only_boxes(MODES, RGB, IMG_SUFFIX, GT, ALL_VGT, LEGT, CLASS_NAMES)
        
        
if __name__ == '__main__':
    count_label_errors(MODES, LEGT)
    #vis_label_errors()