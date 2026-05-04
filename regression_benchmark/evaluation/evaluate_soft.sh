#!/bin/bash

# Evaluation of object detectors with soft labels

#################################################
################# COCO val 2017 #################
#################################################

# Gdino on COCO with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model gdino \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \

# Yolov8 on COCO with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \

# YoloWorld on COCO with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l-worldv2 \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \

# RT-DETR on COCO with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \

# OWL-ViT on COCO with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model owlvit \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \

# Faster R-CNN on COCO with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model frcnn \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val.json \

#################################################
################# Pascal VOC ####################
#################################################

# Gdino on Pascal VOC with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model gdino \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \

# Yolov8 on Pascal VOC with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \

# YoloWorld on Pascal VOC with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l-worldv2 \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \

# RT-DETR on Pascal VOC with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \

# OWL-ViT on Pascal VOC with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model owlvit \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \

# Faster R-CNN on Pascal VOC with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model frcnn \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json \

#################################################
################# KITTI #########################
#################################################

# Gdino on KITTI with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model gdino \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \

# Yolov8 on KITTI with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \

# YoloWorld on KITTI with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l-worldv2 \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \

# RT-DETR on KITTI with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \

# OWL-ViT on KITTI with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model owlvit \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \

# Faster R-CNN on KITTI with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model frcnn \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json \

#################################################
################# Cityscapes val ################
#################################################

# Gdino on Cityscapes with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model gdino \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \

# Yolov8 on Cityscapes with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \

# YoloWorld on Cityscapes with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l-worldv2 \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \

# RT-DETR on Cityscapes with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \

# OWL-ViT on Cityscapes with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model owlvit \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \

# Faster R-CNN on Cityscapes with soft labels
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model frcnn \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json \

##############################################
## Fine tuning on soft vs. hard labels #######
##############################################

# Yolov8 on KITTI (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_hard \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json

# Yolov8 on KITTI (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_soft \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json

# Yolov8 on COCO (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_hard \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json

# Yolov8 on COCO (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_soft \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json

# Yolov8 on Cityscapes (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_hard \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json

# Yolov8 on Cityscapes (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_soft \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json

# Yolov8 on Pascal VOC (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_hard \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json

# Yolov8 on Pascal VOC (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model yolov8l_soft \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json

### RT-DETR

# RT-DETR on KITTI (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_hard \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json

# RT-DETR on KITTI (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_soft \
    --dataset kitti \
    --gt_name kitti_ours \
    --gt_file ../../our_datasets/soft_Kitti_2D_train_val.json

# RT-DETR on COCO (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_hard \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json

# RT-DETR on COCO (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_soft \
    --dataset coco \
    --gt_name coco_ours \
    --gt_file ../../our_datasets/soft_COCO_2017_val_val.json

# RT-DETR on Cityscapes (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_hard \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json

# RT-DETR on Cityscapes (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_soft \
    --dataset cityscapes \
    --gt_name cityscapes_ours \
    --gt_file ../../our_datasets/soft_Cityscapes_val.json

# RT-DETR on Pascal VOC (hard labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_hard \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json

# RT-DETR on Pascal VOC (soft labels)
CUDA_VISIBLE_DEVICES=0 python evaluate_soft.py \
    --model rtdetr-l_soft \
    --dataset pascal \
    --gt_name pascal_ours \
    --gt_file ../../our_datasets/soft_PascalVOC_2012_detection_val.json