#!/bin/bash

#############
# Fine-tune COCO-pretrained models to other datasets with orig. annotations
#############

## KITTI
# RT-DETR fine-tuning on KITTI
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_kitti.py

# YOLOv8 fine-tuning on KITTI
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_kitti.py

# Faster R-CNN fine-tuning on KITTI
CUDA_VISIBLE_DEVICES=0 python fine_tune_frcnn_kitti.py

## Pascal VOC
# RT-DETR fine-tuning on PascalVOC
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_pascal.py

# YOLOv8 fine-tuning on PascalVOC
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_pascal.py

# Faster R-CNN fine-tuning on PascalVOC
CUDA_VISIBLE_DEVICES=0 python fine_tune_frcnn_pascal.py

## Cityscapes
# RT-DETR fine-tuning on Cityscapes
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_cityscapes.py

# YOLOv8 fine-tuning on Cityscapes
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_cityscapes.py

# Faster R-CNN fine-tuning on Cityscapes
CUDA_VISIBLE_DEVICES=0 python fine_tune_frcnn_cityscapes.py

#############
# Fine-tune COCO-pretrained models on our annotations with hard labels (threshold 0.5)
#############

## Faster R-CNN
# Faster R-CNN fine-tuning on KITTI with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_frcnn_kitti_ours_hard.py

# Faster R-CNN fine-tuning on Cityscapes with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_frcnn_cityscapes_ours_hard.py

# Faster R-CNN fine-tuning on PascalVOC with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_frcnn_pascal_ours_hard.py

# Faster R-CNN fine-tuning on COCO with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_frcnn_coco_ours_hard.py

## RT-DETR
# RT-DETR fine-tuning on KITTI with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_kitti_ours_hard.py

# RT-DETR fine-tuning on PascalVOC with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_pascal_ours_hard.py

# RT-DETR fine-tuning on Cityscapes with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_cityscapes_ours_hard.py

# RT-DETR fine-tuning on COCO with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_coco_ours_hard.py

## YOLOv8
# YOLOv8 fine-tuning on KITTI with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_kitti_ours_hard.py

# YOLOv8 fine-tuning on PascalVOC with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_pascal_ours_hard.py

# YOLOv8 fine-tuning on Cityscapes with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_cityscapes_ours_hard.py

# YOLOv8 fine-tuning on COCO with our hard labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_coco_ours_hard.py

#############
# Fine-tune COCO-pretrained models on our annotations with soft labels
#############

## YOLOv8
# YOLOv8 fine-tuning on KITTI with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_kitti_ours_soft.py

# YOLOv8 fine-tuning on Cityscapes with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_cityscapes_ours_soft.py

# YOLOv8 fine-tuning on COCO with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_coco_ours_soft.py

# YOLOv8 fine-tuning on PascalVOC with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_yolov8l_pascal_ours_soft.py

## RT-DETR
# RT-DETR fine-tuning on KITTI with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_kitti_ours_soft.py

# RT-DETR fine-tuning on PascalVOC with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_pascal_ours_soft.py

# RT-DETR fine-tuning on Cityscapes with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_cityscapes_ours_soft.py

# RT-DETR fine-tuning on COCO with our soft labels
CUDA_VISIBLE_DEVICES=0 python fine_tune_rtdetr_coco_ours_soft.py