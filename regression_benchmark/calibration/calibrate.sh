#!/bin/bash

# Post-hoc Calibration of Models using Isotonic Regression through k-fold Cross-Validation on the Validation Set

#
# Grounding DINO
#

# Grounding DINO on COCO
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model gdino \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# Grounding DINO on Cityscapes
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model gdino \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

# Grounding DINO on KITTI
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model gdino \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# Grounding DINO on Pascal VOC
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model gdino \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

#
# YOLOv8
#

# YOLOv8 on COCO
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model yolov8l \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# YOLOv8 on Cityscapes
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model yolov8l \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

# YOLOv8 on KITTI
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model yolov8l \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# YOLOv8 on Pascal VOC
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model yolov8l \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

#
# Yolo World
#

# Yolo World on COCO
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model yolov8l-worldv2 \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# Yolo World on Cityscapes
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model yolov8l-worldv2 \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

# Yolo World on KITTI
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model yolov8l-worldv2 \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# Yolo World on Pascal VOC
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model yolov8l-worldv2 \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

#
# RT-DETR
#

# RT-DETR on COCO
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model rtdetr-l \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# RT-DETR on Cityscapes
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model rtdetr-l \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

# RT-DETR on KITTI
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model rtdetr-l \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# RT-DETR on Pascal VOC
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model rtdetr-l \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

#
# OWL-ViT
#

# OWL-ViT on COCO
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model owlvit \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# OWL-ViT on Cityscapes
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model owlvit \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

# OWL-ViT on KITTI
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model owlvit \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# OWL-ViT on Pascal VOC
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model owlvit \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

#
# Faster R-CNN
#

# Faster R-CNN on COCO
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model frcnn \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# Faster R-CNN on Cityscapes
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model frcnn \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

# Faster R-CNN on KITTI
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model frcnn \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# Faster R-CNN on Pascal VOC
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model frcnn \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

#
# YOLOv8 trained on hard vs. soft labels
#


# YOLOv8 on KITTI (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model yolov8l_hard \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# YOLOv8 on KITTI (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model yolov8l_soft \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize

# YOLOv8 on COCO (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model yolov8l_hard \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# YOLOv8 on COCO (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model yolov8l_soft \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize

# YOLOv8 on Pascal VOC (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model yolov8l_hard \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

# YOLOv8 on Pascal VOC (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model yolov8l_soft \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

# YOLOv8 on Cityscapes (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model yolov8l_hard \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

# YOLOv8 on Cityscapes (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model yolov8l_soft \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize

#
# RT-DETR trained on hard vs. soft labels
#

# RT-DETR on KITTI (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model rtdetr-l_hard \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize 

# RT-DETR on KITTI (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset kitti \
    --model rtdetr-l_soft \
    --gt_file /path/to/datasets/KITTI/val/instances_val.json \
    --visualize 

# RT-DETR on COCO (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model rtdetr-l_hard \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize 

# RT-DETR on COCO (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset coco \
    --model rtdetr-l_soft \
    --gt_file /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --visualize 

# RT-DETR on Pascal VOC (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model rtdetr-l_hard \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize 

# RT-DETR on Pascal VOC (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset pascal \
    --model rtdetr-l_soft \
    --gt_file /path/to/datasets/VOC/labels/instances_val2012.json \
    --visualize

# RT-DETR on Cityscapes (hard labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model rtdetr-l_hard \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize 

# RT-DETR on Cityscapes (soft labels)
CUDA_VISIBLE_DEVICES=0 python calibrate.py \
    --dataset cityscapes \
    --model rtdetr-l_soft \
    --gt_file /path/to/datasets/Cityscapes/val/instances_val.json \
    --visualize
