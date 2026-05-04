import argparse
import os
import sys
import time
import json
from tqdm import tqdm
import cv2
import torch
import numpy as np

from ultralytics import YOLO

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from ultralytics.engine.results import Results

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel
import torch.nn.functional as F

from typing import Any
from ultralytics.utils.tal import make_anchors

# Patch Ultralytics to support training with soft labels

class SoftYOLODataset(YOLODataset):

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # already contains cls and bboxes fields from parent class, we add soft labels as new field

        # get soft labels from file
        label_path = self.label_files[idx]
        soft_label_path = label_path.replace("labels", "soft_labels")  # separate folder to not break internal ultralytics label checks

        soft_cls = self.load_soft_labels(soft_label_path) # load soft labels for this image

        sample["soft_cls"] = soft_cls  # add soft label as new field

        return sample
    
    def load_soft_labels(self, label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        # extract only soft label parts
        soft = [list(map(float, line.strip().split()[5:])) for line in lines]

        return torch.tensor(soft, dtype=torch.float32)


    def collate_fn(self, batch):
        out = super().collate_fn(batch)

        if "soft_cls" in batch[0]:
            out["soft_cls"] = torch.cat([b["soft_cls"] for b in batch], 0)

        return out
    

class SoftDetectionModel(DetectionModel):

    def init_criterion(self):
        return SoftDetectionLoss(self)
    
class SoftDetectionLoss(v8DetectionLoss):

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        if "soft_cls" in batch:

            soft = batch["soft_cls"].to(target_scores.device).float()

            soft_target_scores = target_scores.clone()

            # only use soft label information for foreground anchors
            soft_target_scores[fg_mask] = soft[target_gt_idx[fg_mask]]


        target_scores_sum = max(target_scores.sum(), 1)

        
        # Updated classification loss with soft labels
        if "soft_cls" in batch:

            orig_bce_loss = self.bce(pred_scores, target_scores.to(dtype))
            soft_bce_loss = self.bce(pred_scores, soft_target_scores.to(dtype))  # (bs, num_anchors, nc)

            # print(orig_bce_loss.sum() / target_scores_sum)

            # print(soft_bce_loss.sum() / target_scores_sum)

            loss[1] = orig_bce_loss.sum() / target_scores_sum + 0.1 * (soft_bce_loss.sum() / target_scores_sum) # BCE with soft labels

        else: # original hard BCE loss
            bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
            if self.class_weights is not None: # not needed by us and disabled by default
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / target_scores_sum  # BCE


        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def __call__(self, preds, batch):
        
        # print(">>> USING SOFT LOSS <<<")

        loss, loss_items = super().__call__(preds, batch)

        return loss, loss_items

class SoftYOLOTrainer(DetectionTrainer):

    def build_dataset(self, img_path, mode="train", batch=None):

        return SoftYOLODataset(
            data=self.data,              
            task=self.args.task,
            img_path=img_path,           
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=False,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=int(self.stride),
            prefix=f"{mode}: ",
        )
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SoftDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)

        if weights:
            model.load(weights)

        return model
    
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
        "YOLOv8 eval on PascalVOC", add_help=True)

    # PascalVOC image directory
    parser.add_argument("--image_dir", type=str,
                        required=True, help="pascalvoc image dir")

    parser.add_argument("--save_dir", type=str, default="predictions/yolov8l/pascalvoc/", help="directory to save predictions")
    parser.add_argument("--checkpoint_path", type=str, default="../training/runs/detect/yolov8l_pascalvoc/weights/best.pt", help="path to the model checkpoint")
    args = parser.parse_args()

    id_map = { i : i for i in range(20) } # Identity mapping for PascalVOC with 20 classes, mapping only required for COCO
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load our pre-trained YOLOv8 model
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found at {args.checkpoint_path}. Make sure to train the model first and provide the correct path.")
        sys.exit(1)
    model = YOLO(args.checkpoint_path)

    # Apply Patch
    model.predictor = CustomYOLOPredictor(overrides=model.overrides)
    model.predictor.setup_model(model=model.model)

    # Inference on PascalVOC
    results = model.predict(
        source = args.image_dir,
        conf = 0.001,
        iou = 0.7,
        max_det = 300,
        stream=False # we have to store ourselves
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