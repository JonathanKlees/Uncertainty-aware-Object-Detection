import argparse
import os
import sys
import time
import json
from tqdm import tqdm
import cv2
import torch

###
# NOTE: Patches Ultralytics RT-DETR Post-processing to return class probabilities. 
# Patch was written for Ultralytics 8.4.22 (Python 3.10.20), so you may need to adjust imports for the latest version.
###

from ultralytics import RTDETR
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

class CustomRTDETRPredictor(RTDETRPredictor):

    # Patch Post-Processing for RT-DETR

    def postprocess(self, preds, img, orig_imgs):

        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []

        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):

            # --- store full probabilities ---
            probs = score   # [num_queries, num_classes]

            # --- existing code ---
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)

            idx = max_score.squeeze(-1) > self.args.conf

            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx

            # --- apply SAME filtering ---
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]
            probs = probs[idx]   # keep aligned with pred

            # --- sort & limit to max detections ---
            order = pred[:, 4].argsort(descending=True)
            pred = pred[order][: self.args.max_det]
            probs = probs[order][: self.args.max_det]   # same ordering

            # --- scale boxes ---
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh

            # --- build result ---
            r = Results(orig_img, path=img_path, names=self.model.names, boxes=pred)

            # --- attach probabilities ---
            r.class_probs = probs.cpu().numpy()

            results.append(r)

        return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "RT-DETR eval on Cityscapes", add_help=True)

    # Cityscapes image directory
    parser.add_argument("--image_dir", type=str,
                        required=True, help="cityscapes image dir")

    parser.add_argument("--save_dir", type=str, default="predictions/rtdetr-l/cityscapes/", help="directory to save predictions")

    parser.add_argument("--checkpoint_path", type=str, default="../training/runs/detect/rtdetr-l_cityscapes/weights/best.pt", help="path to the model checkpoint")


    args = parser.parse_args()

    id_map = { i : i for i in range(8) } # Identity mapping for Cityscapes with 8 classes, mapping only required for COCO


    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load COCO pre-trained RT-DETR model
    model =  RTDETR(args.checkpoint_path)

    # Replace predictor
    model.predictor = CustomRTDETRPredictor(overrides=model.overrides)
    model.predictor.setup_model(model=model.model)

    # Inference on COCO
    results = model.predict(
        source = args.image_dir,
        conf = 0.001,
        iou = 0.7,
        max_det = 300,
        stream=False # we have to store ourselves
    )

    # Store the predictions in COCO format (one json per image)
    for r in tqdm(results):
        image_id = os.path.basename(r.path).split(".")[0]

        if r.boxes is None:
            boxes, scores, labels, probs = [], [], [], []
        else:
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            scores = r.boxes.conf.cpu().numpy().tolist()
            labels = r.boxes.cls.cpu().numpy().tolist()
            probs = r.class_probs.tolist()

        save_dict = {
            "image_id": image_id,
            "boxes": boxes,
            "scores": scores,
            "labels": [id_map[int(label)] for label in labels], # map to COCO category IDs
            "probs": probs,
        }

        with open(os.path.join(save_dir, f"{image_id}.json"), "w") as f:
            json.dump(save_dict, f, indent = 2)