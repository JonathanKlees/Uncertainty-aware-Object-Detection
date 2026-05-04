from helpers_vis import *
from helpers_legt import *

#########################################################################
DATASET = "Cityscapes"
MODES = ["train", "val"]
IMG_SUFFIX = ".png"
CLASS_NAMES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
ERROR_TYPES = ["missing", "misaligned", "classification", "original_only"]
GT = f"/path/to/benchmark_datasets_gt/{DATASET}"
VGT = f"/path/to/benchmark_datasets_vgt/{DATASET}"
ALL_VGT = f"/path/to/benchmark_datasets_vgt_all_bboxes/{DATASET}"
LEGT = f"/path/to/benchmark_datasets_legt_second_round/{DATASET}"
#########################################################################


def vis_label_errors():
    vis_main_errors(MODES, GT, IMG_SUFFIX, GT, VGT, LEGT, CLASS_NAMES)
    vis_original_only(MODES, GT, IMG_SUFFIX, GT, ALL_VGT, LEGT, CLASS_NAMES)
    vis_random_original_only_boxes(MODES, GT, IMG_SUFFIX, GT, ALL_VGT, LEGT, CLASS_NAMES)
        
        
if __name__ == '__main__':
    count_label_errors(MODES, LEGT)
    #vis_label_errors()