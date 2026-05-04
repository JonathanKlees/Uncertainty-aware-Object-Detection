from helpers_vis import *
from helpers_legt import *

#########################################################################
DATASET = "PascalVOC"
MODES = ["train", "val"]
IMG_SUFFIX = ".jpg"
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
ERROR_TYPES = ["missing", "misaligned", "classification", "original_only"]
GT = f"/path/to/benchmark_datasets_gt/{DATASET}"
VGT = f"/path/to/benchmark_datasets_vgt/{DATASET}"
ALL_VGT = f"/path/to/benchmark_datasets_vgt_all_bboxes/{DATASET}"
LEGT = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}"
#########################################################################


def vis_label_errors():
    vis_main_errors(MODES, GT, IMG_SUFFIX, GT, VGT, LEGT, CLASS_NAMES)
    vis_original_only(MODES, GT, IMG_SUFFIX, GT, ALL_VGT, LEGT, CLASS_NAMES)
    vis_random_original_only_boxes(modes=MODES, rgb=GT, img_suffix=IMG_SUFFIX, gt=GT, all_vgt=ALL_VGT, legt=LEGT, class_names=CLASS_NAMES)
        
        
if __name__ == '__main__':
    count_label_errors(MODES, LEGT)
    #vis_label_errors()