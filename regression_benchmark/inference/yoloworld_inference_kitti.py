import argparse
import os
import sys
import time
import json
from tqdm import tqdm
import cv2
import torch
import numpy as np

from ultralytics import YOLOWorld

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from ultralytics.engine.results import Results
###
# NOTE: Patches Ultralytics NMS function in YOLO Post-processing to return class probabilities. 
# Patch was written for Ultralytics 8.4.22 (Python 3.10.20), so you may need to adjust imports for the latest version.
###

class CustomYOLOPredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_imgs):

        preds_nms, indices = non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            max_det=self.args.max_det,
            return_idxs=True
        )

        results = []

        for i, (pred, idx) in enumerate(zip(preds_nms, indices)):
            orig_img = orig_imgs[i]
            path = self.batch[0][i]

            if pred is None or len(pred) == 0:
                empty_boxes = torch.zeros((0, 6), device=img.device)

                r = Results(
                    orig_img=orig_img,
                    path=path,
                    names=self.model.names,
                    boxes=empty_boxes
                )

                r.class_probs = np.zeros((0, len(self.model.names)))

            else:
                # Scale boxes back to original image size
                # pred[:, :4] = x1, y1, x2, y2 in model input size
                pred[:, :4] = scale_boxes(
                    img.shape[2:],  # model input H,W
                    pred[:, :4],    # boxes in model input
                    orig_img.shape[:2]  # original image H,W
                ).round()

                r = Results(
                    orig_img=orig_img,
                    path=path,
                    names=self.model.names,
                    boxes=pred
                )

                raw = preds[i][0]              # [C, N]
                raw = raw[:, idx.long()]      # [C, n]
                raw = raw.T                  # [n, C]

                r.class_probs = raw[:, 4:].cpu().numpy()

            results.append(r)

        return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "YOLOv8 eval on Kitti", add_help=True)

    # Kitti image directory
    parser.add_argument("--image_dir", type=str,
                        required=True, help="kitti image dir")
    
    parser.add_argument("--model_name", type=str, default="yolov8l-worldv2", help="name of the model")

    parser.add_argument("--save_dir", type=str, default="predictions/yolov8l-worldv2/kitti/", help="directory to save predictions")

    args = parser.parse_args()

    id_map = {i : i for i in range(7)} # Identity mapping for the 7 classes in Kitti, map is only required for COCO

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load COCO pre-trained YOLOv8 model
    model = YOLOWorld(f"checkpoints/{args.model_name}.pt")
    model.set_classes(["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram"])

    # Apply Patch

    model.predictor = CustomYOLOPredictor(overrides=model.overrides)
    model.predictor.setup_model(model=model.model)

    # Inference on Kitti
    results = model.predict(
        source = args.image_dir,
        conf = 0.001,
        iou = 0.7,
        max_det = 300,
        stream=False, # we store ourselves in COCO format
    )

    # Store the predictions in COCO format (one json per image)
    for r in tqdm(results):
        image_id = int(os.path.basename(r.path).split(".")[0])

        if r.boxes is None:
            boxes, scores, labels, probs = [], [], [], []
        else:
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            scores = r.boxes.conf.cpu().numpy().tolist()
            labels = r.boxes.cls.flatten().cpu().numpy().tolist()
            probs = r.class_probs.tolist() # new attribute from the patch

        save_dict = {
            "image_id": image_id,
            "boxes": boxes,
            "scores": scores,
            "labels": [id_map[int(label)] for label in labels],
            "probs": probs,
        }

        with open(os.path.join(save_dir, f"{image_id}.json"), "w") as f:
            json.dump(save_dict, f, indent = 2)