#!/bin/bash

### Grounding DINO

# grounding dino inference on coco
CUDA_VISIBLE_DEVICES=0 python gdino_inference_coco.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --anno_path /path/to/datasets/COCO/2017/annotations/instances_val2017.json \
    --image_dir /path/to/datasets/COCO/2017/val2017 \
    --save_dir predictions/gdino/coco/

# grounding dino inference on cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python gdino_inference_cityscapes.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --label_dir /path/to/datasets/Cityscapes/val/labels \
    --image_dir /path/to/datasets/Cityscapes/val/images

# grounding dino inference on pascalVOC validation data
CUDA_VISIBLE_DEVICES=0 python gdino_inference_pascal.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --label_dir /path/to/datasets/VOC/labels/val2012 \
    --image_dir /path/to/datasets/VOC/images/val2012

# grounding dino inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python gdino_inference_kitti.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --label_dir /path/to/datasets/KITTI/val/labels \
    --image_dir /path/to/datasets/KITTI/val/images


### YOLO-World


# YoloWorld inference on coco
CUDA_VISIBLE_DEVICES=0 python yoloworld_inference_coco.py \
    --image_dir /path/to/datasets/COCO/2017/val2017

# YoloWorld inference on cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python yoloworld_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images

# YoloWorld inference on Kitti custom validation data
CUDA_VISIBLE_DEVICES=0 python yoloworld_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images

# YoloWorld inference on PascalVOC validation data
CUDA_VISIBLE_DEVICES=0 python yoloworld_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012


### OWL-ViT


# Owl-ViT inference on coco
CUDA_VISIBLE_DEVICES=0 python owlvit_inference_coco.py \
    --image_dir /path/to/datasets/COCO/2017/val2017

# Owl-ViT inference on cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python owlvit_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images

# Owl-ViT inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python owlvit_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images

# Owl-ViT inference on PascalVOC validation data
CUDA_VISIBLE_DEVICES=0 python owlvit_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012


### Faster R-CNN


# Faster R-CNN inference on coco
CUDA_VISIBLE_DEVICES=0 python frcnn_inference_coco.py \
    --image_dir /path/to/datasets/COCO/2017/val2017

# Faster-RCNN inference on Cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python frcnn_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images

# Faster-RCNN inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python frcnn_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images

# Faster-RCNN inference on PascalVOC validation data
CUDA_VISIBLE_DEVICES=0 python frcnn_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012


### YOLOv8


# yolov8 inference on coco
CUDA_VISIBLE_DEVICES=0 python yolov8_inference_coco.py \
    --image_dir /path/to/datasets/COCO/2017/val2017

# YOLOv8 inference on Cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images

# YOLOv8 inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images

# YOLOv8 inference on PascalVOC validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012 


### RT-DETR


# RT-DETR inference on coco
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_coco.py \
    --image_dir /path/to/datasets/COCO/2017/val2017

# RT-DETR inference on Cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images

# RT-DETR inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images

# RT-DETR inference on PascalVOC validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012 


#######################

# additional inference for fine-tuned models on hard vs. soft labels

# make sure that fine-tuned models exist by running the fine-tuning script first (fine_tune.sh) before running this inference script
# otherwise the checkpoints won't be found and the script will throw an error.


### YOLOv8

## KITTI
# YOLOv8 trained on hard labels inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images \
    --save_dir "predictions/yolov8l_hard/kitti/" \
    --checkpoint_path "../training/runs/detect/yolov8l_kitti_ours_hard/weights/best.pt"
# YOLOv8 trained on soft labels inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_soft_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images \
    --save_dir "predictions/yolov8l_soft/kitti/" \
    --checkpoint_path "../training/runs/detect/yolov8l_kitti_ours_soft/weights/best.pt"

## COCO
# YOLOv8 trained with hard labels inference on COCO custom validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_soft_inference_coco.py \
    --image_dir /path/to/datasets/COCO_ours/val/images/ \
    --save_dir "predictions/yolov8l_hard/coco/" \
    --checkpoint_path "../training/runs/detect/yolov8l_coco_ours_hard/weights/best.pt" 
# YOLOv8 trained with soft labels inference on COCO custom validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_soft_inference_coco.py \
    --image_dir /path/to/datasets/COCO_ours/val/images/ \
    --save_dir "predictions/yolov8l_soft/coco/" \
    --checkpoint_path "../training/runs/detect/yolov8l_coco_ours_soft/weights/best.pt"

## Pascal VOC
# YOLOv8 trained with hard labels inference on VOC validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_soft_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012 \
    --save_dir "predictions/yolov8l_hard/pascal/" \
    --checkpoint_path "../training/runs/detect/yolov8l_pascal_ours_hard/weights/best.pt"
YOLOv8 trained with soft labels inference on VOC validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_soft_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012 \
    --save_dir "predictions/yolov8l_soft/pascal/" \
    --checkpoint_path "../training/runs/detect/yolov8l_pascal_ours_soft/weights/best.pt"

## Cityscapes
# YOLOv8 trained with hard labels inference on Cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_soft_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images \
    --save_dir "predictions/yolov8l_hard/cityscapes/" \
    --checkpoint_path "../training/runs/detect/yolov8l_cityscapes_ours_hard/weights/best.pt"
# YOLOv8 trained with soft labels inference on Cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python yolov8_soft_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images \
    --save_dir "predictions/yolov8l_soft/cityscapes/" \
    --checkpoint_path "../training/runs/detect/yolov8l_cityscapes_ours_soft/weights/best.pt"


### RT-DETR

## KITTI
# RT-DETR trained on hard labels inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images \
    --save_dir "predictions/rtdetr-l_hard/kitti/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_kitti_ours_hard/weights/best.pt"

# RT-DETR trained on soft labels inference on KITTI custom validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_soft_inference_kitti.py \
    --image_dir /path/to/datasets/KITTI/val/images \
    --save_dir "predictions/rtdetr-l_soft/kitti/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_kitti_ours_soft/weights/best.pt"

## COCO
# RT-DETR trained with hard labels inference on COCO custom validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_coco.py \
    --image_dir /path/to/datasets/COCO_ours/val/images/ \
    --save_dir "predictions/rtdetr-l_hard/coco/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_coco_ours_hard/weights/best.pt" 

# RT-DETR trained with soft labels inference on COCO custom validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_soft_inference_coco.py \
    --image_dir /path/to/datasets/COCO_ours/val/images/ \
    --save_dir "predictions/rtdetr-l_soft/coco/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_coco_ours_soft/weights/best.pt"

## Pascal VOC
# RT-DETR trained with hard labels inference on VOC validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012 \
    --save_dir "predictions/rtdetr-l_hard/pascal/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_pascal_ours_hard/weights/best.pt"

# RT-DETR trained with soft labels inference on VOC validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_soft_inference_pascal.py \
    --image_dir /path/to/datasets/VOC/images/val2012 \
    --save_dir "predictions/rtdetr-l_soft/pascal/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_pascal_ours_soft/weights/best.pt"

## Cityscapes
# RT-DETR trained with hard labels inference on Cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images \
    --save_dir "predictions/rtdetr-l_hard/cityscapes/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_cityscapes_ours_hard/weights/best.pt"

# RT-DETR trained with soft labels inference on Cityscapes validation data
CUDA_VISIBLE_DEVICES=0 python rtdetr_soft_inference_cityscapes.py \
    --image_dir /path/to/datasets/Cityscapes/val/images \
    --save_dir "predictions/rtdetr-l_soft/cityscapes/" \
    --checkpoint_path "../training/runs/detect/rtdetr-l_cityscapes_ours_soft/weights/best.pt"


###
# Important: for the Cityscapes dataset, after running the inference scripts,
# you need to run the following script to convert the filenames of the predictions to the corresponding IDs.
###
python convert_cityscapes_filenames_to_id.py
