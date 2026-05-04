#!/bin/bash

######################################################
###############  COCO val 2017 #######################
######################################################

#
# Grounding DINO
#

# gdino on orig. COCO 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset coco \
    --gt_name coco_orig_gt \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json

# gdino on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# gdino on COCO with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset coco \
    --gt_name coco_ours_0.8 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YOLOv8
#

# yolov8l on orig. COCO 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset coco \
    --gt_name coco_orig_gt \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json

# yolov8l on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l on COCO with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset coco \
    --gt_name coco_ours_0.8 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YoloWorld
#

# YoloWorld on orig. COCO 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset coco \
    --gt_name coco_orig_gt \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json

# YoloWorld on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# YoloWorld on COCO with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset coco \
    --gt_name coco_ours_0.8 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# RT-DETR
#

# RT-DETR on orig. COCO 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset coco \
    --gt_name coco_orig_gt \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json

# RT-DETR on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR on COCO with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset coco \
    --gt_name coco_ours_0.8 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# OwlViT
#

# OwlViT on orig. COCO 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset coco \
    --gt_name coco_orig_gt \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json

# OwlViT on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# OwlViT on COCO with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset coco \
    --gt_name coco_ours_0.8 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# Faster R-CNN
#

# Faster R-CNN on orig. COCO 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset coco \
    --gt_name coco_orig_gt \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json

# Faster R-CNN on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# Faster R-CNN on COCO with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset coco \
    --gt_name coco_ours_0.8 \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

######################################################
###############  Pascal VOC 2012 #####################
######################################################

#
# Grounding DINO
#

# gdino on orig. VOC
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset pascal \
    --gt_name pascal_orig_gt \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json

# gdino on VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# gdino on VOC with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset pascal \
    --gt_name pascal_ours_0.8 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YOLOv8
#

# yolov8l on orig. VOC 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset pascal \
    --gt_name pascal_orig_gt \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json

# yolov8l on VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l on VOC with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset pascal \
    --gt_name pascal_ours_0.8 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YoloWorld
#

# YoloWorld on orig. VOC 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset pascal \
    --gt_name pascal_orig_gt \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json

# YoloWorld on VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# YoloWorld on VOC with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset pascal \
    --gt_name pascal_ours_0.8 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# RT-DETR
#

# RT-DETR on orig. VOC 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset pascal \
    --gt_name pascal_orig_gt \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json

# RT-DETR on VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR on VOC with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset pascal \
    --gt_name pascal_ours_0.8 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# OwlViT
#

# OwlViT on orig. VOC
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset pascal \
    --gt_name pascal_orig_gt \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json

# OwlViT on VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# OwlViT on VOC with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset pascal \
    --gt_name pascal_ours_0.8 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# Faster R-CNN
#

# Faster R-CNN on orig. VOC
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset pascal \
    --gt_name pascal_orig_gt \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json

# Faster R-CNN on VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# Faster R-CNN on VOC with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset pascal \
    --gt_name pascal_ours_0.8 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

######################################################
########### Cityscapes ###########
######################################################

#
# Grounding DINO
#

# gdino on orig. Cityscapes
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset cityscapes \
    --gt_name cityscapes_orig_gt \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json

# gdino on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# gdino on Cityscapes with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.8 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YOLOv8
#

# yolov8l on orig. Cityscapes 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset cityscapes \
    --gt_name cityscapes_orig_gt \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json

# yolov8l on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l on Cityscapes with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.8 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YoloWorld
#

# YoloWorld on orig. Cityscapes
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset cityscapes \
    --gt_name cityscapes_orig_gt \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json

# YoloWorld on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# YoloWorld on Cityscapes with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.8 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# RT-DETR
#

# RT-DETR on orig. Cityscapes 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset cityscapes \
    --gt_name cityscapes_orig_gt \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json

# RT-DETR on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR on Cityscapes with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.8 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# OwlViT
#

# OwlViT on orig. Cityscapes
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset cityscapes \
    --gt_name cityscapes_orig_gt \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json

# OwlViT on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# OwlViT on Cityscapes with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.8 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# Faster R-CNN
#

# Faster R-CNN on orig. Cityscapes
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset cityscapes \
    --gt_name cityscapes_orig_gt \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json

# Faster R-CNN on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# Faster R-CNN on Cityscapes with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.8 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

######################################################
################### KITTI 2D #########################
######################################################

#
# Grounding DINO
#

# gdino on orig. KITTI
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset kitti \
    --gt_name kitti_orig_gt \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json

# gdino on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# gdino on KITTI with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model gdino \
    --dataset kitti \
    --gt_name kitti_ours_0.8 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YOLOv8
#

# yolov8l on orig. KITTI 
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset kitti \
    --gt_name kitti_orig_gt \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json

# yolov8l on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l on KITTI with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l \
    --dataset kitti \
    --gt_name kitti_ours_0.8 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# YoloWorld
#

# YoloWorld on orig. KITTI
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset kitti \
    --gt_name kitti_orig_gt \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json

# YoloWorld on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# YoloWorld on KITTI with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l-worldv2 \
    --dataset kitti \
    --gt_name kitti_ours_0.8 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# RT-DETR
#

# RT-DETR on orig. KITTI
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset kitti \
    --gt_name kitti_orig_gt \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json

# RT-DETR on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR on KITTI with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l \
    --dataset kitti \
    --gt_name kitti_ours_0.8 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# OwlViT
#

# OwlViT on orig. KITTI
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset kitti \
    --gt_name kitti_orig_gt \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json

# OwlViT on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# OwlViT on KITTI with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model owlvit \
    --dataset kitti \
    --gt_name kitti_ours_0.8 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

#
# Faster R-CNN
#

# Faster R-CNN on orig. KITTI
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset kitti \
    --gt_name kitti_orig_gt \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json

# Faster R-CNN on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# Faster R-CNN on KITTI with p > 0.8
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model frcnn \
    --dataset kitti \
    --gt_name kitti_ours_0.8 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.8

###################################################

# yolov8l trained on hard labels on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_hard \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l trained on soft labels on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_soft \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l trained on hard labels on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_hard \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l trained on soft labels on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_soft \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l trained on hard labels on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_hard \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l trained on soft labels on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_soft \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l trained on hard labels on Pascal VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_hard \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# yolov8l trained on soft labels on Pascal VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model yolov8l_soft \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5


### RT-DETR

# RT-DETR trained on hard labels on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_hard \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR trained on soft labels on KITTI with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_soft \
    --dataset kitti \
    --gt_name kitti_ours_0.5 \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR trained on hard labels on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_hard \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR trained on soft labels on COCO with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_soft \
    --dataset coco \
    --gt_name coco_ours_0.5 \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR trained on hard labels on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_hard \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR trained on soft labels on Cityscapes with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_soft \
    --dataset cityscapes \
    --gt_name cityscapes_ours_0.5 \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR trained on hard labels on Pascal VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_hard \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5

# RT-DETR trained on soft labels on Pascal VOC with p > 0.5
CUDA_VISIBLE_DEVICES=0 python evaluate_hard.py \
    --model rtdetr-l_soft \
    --dataset pascal \
    --gt_name pascal_ours_0.5 \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \
    --soft_labels_available \
    --soft_label_thresh 0.5