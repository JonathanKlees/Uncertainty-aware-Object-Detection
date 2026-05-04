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

### Patches for soft label training

import torch
from ultralytics.models.rtdetr.val import RTDETRDataset

class SoftRTDETRDataset(RTDETRDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        label_path = self.label_files[idx]
        sample["dataset_idx"] = idx # store index for later retrieval in loss
        soft_path = label_path.replace("labels", "soft_labels")

        sample["soft_cls"] = self.load_soft_labels(soft_path)

        return sample

    def load_soft_labels(self, path):
        try:
            with open(path, "r") as f:
                lines = f.readlines()

            soft = [
                list(map(float, line.strip().split()[5:]))
                for line in lines
            ]

            return torch.tensor(soft, dtype=torch.float32)

        except FileNotFoundError:
            return torch.zeros((0, self.data["nc"]), dtype=torch.float32)

    def collate_fn(self, batch):
        out = super().collate_fn(batch)

        if "soft_cls" in batch[0]:
            out["soft_cls"] = [b["soft_cls"] for b in batch]

        if "dataset_idx" in batch[0]:
            out["dataset_idx"] = [b["dataset_idx"] for b in batch]

        return out
    

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import colorstr

class SoftRTDETRTrainer(RTDETRTrainer):

    def build_dataset(self, img_path, mode="train", batch=None):
        self.dataset = SoftRTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
        return self.dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SoftRTDETRDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)

        if weights:
            model.load(weights)

        return model
    
from ultralytics.nn.tasks import RTDETRDetectionModel

class SoftRTDETRDetectionModel(RTDETRDetectionModel):

    # patch loss such that information flows through
    def loss(self, batch, preds=None):
        """Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (tuple, optional): Precomputed model predictions.

        Returns:
            (torch.Tensor): Total loss value.
            (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "soft_cls": batch["soft_cls"] if "soft_cls" in batch else None,
            "im_file": batch["im_file"],
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

import ultralytics.models.utils.loss as loss_module
import torch
import torch.nn.functional as F
import torch.nn as nn

OriginalLossClass = loss_module.RTDETRDetectionLoss._get_loss_class

def patched_get_loss_class(
    self,
    pred_scores,
    targets,
    gt_scores,
    num_gts,
    postfix=""
):
    name_class = f"loss_class{postfix}"
    device = pred_scores.device
    bs, nq, nc = pred_scores.shape

    # ----------------------------
    # ORIGINAL LOSS (unchanged for reference)
    # ----------------------------
    
    one_hot = torch.zeros((bs, nq, nc + 1), dtype=torch.int64, device=targets.device)
    one_hot.scatter_(2, targets.unsqueeze(-1), 1)
    one_hot = one_hot[..., :-1]

    # print("First few targets:", targets[0,:100])
    # Targets contain background class for non-matched predictions, removed in the last line.

    # print("GT Scores shape:", gt_scores.shape)

    hard_gt_scores = gt_scores.view(bs, nq, 1) * one_hot

    # gt scores are IoU values for the nq predictions

    # print("GT Scores after one-hot shape:", hard_gt_scores.shape)

    if self.fl:
        if num_gts and self.vfl:
            loss_cls = self.vfl(pred_scores, hard_gt_scores, one_hot)
        else:
            loss_cls = self.fl(pred_scores, one_hot.float())
        loss_cls /= max(num_gts, 1) / nq
    else:
        loss_cls = torch.nn.BCEWithLogitsLoss(reduction="none")(pred_scores, hard_gt_scores).mean(1).sum()

    # ----------------------------
    # SOFT LABEL LOSS (new code for soft labels)
    # ----------------------------

    # ----------------------------
    # 1. ACCESS STORED DATA
    # ----------------------------

    batch = getattr(self, "_soft_batch", None)

    # print(f"Batch keys: {batch.keys() if batch is not None else 'No batch'}")

    match_indices = getattr(self, "_soft_match_indices", None)

    if batch is None or match_indices is None:
        return {name_class: loss_cls * self.loss_gain["class"]}

    # ----------------------------
    # SOFT LABEL Alignment (Some hard labels are filtered)
    # ----------------------------
    
    # print(batch["cls"].shape, batch["soft_cls"].shape) # always some more soft labels due to internal filtering

    # Workaround: Load from filename and match to hard labels
    batch_soft_labels = []
    for i in range(bs): # align per image
        hard_labels = batch["cls"][batch["batch_idx"] == i]  # (N_hard,) hard labels for this image
        soft_labels = batch["soft_cls"][i].to(targets.device)  # (N_soft, num_classes) soft labels for this image
       
        # print(hard_labels.shape, soft_labels.shape)

        if hard_labels.shape[0] < soft_labels.shape[0]:
            # Some annotations are dropped along the way, but we can still map soft labels to hard labels since order is preserved.
            soft_labels = self.align_soft_labels_to_hard_labels(soft_labels, hard_labels)

            # print("After alignment:", hard_labels.shape, soft_labels.shape)
            # print(hard_labels, soft_labels.argmax(dim=1)) # check if alignment worked correctly

        if hard_labels.shape[0] != soft_labels.shape[0]:
            print(f"Warning: Image {batch['im_file'][i]} has {hard_labels.shape[0]} hard labels but {soft_labels.shape[0]} soft labels after alignment. This may indicate a mismatch in annotations or an issue with the alignment process.")

            print("Hard labels:", hard_labels.long())
            print("Soft labels:", soft_labels.argmax(dim=1))
            print("Bboxes:", batch["bboxes"][batch["batch_idx"] == i])
        batch_soft_labels.append(soft_labels)

    # ----------------------------
    # Build soft target tensor aligned with pred_scores using match_indices
    # ----------------------------

    soft_targets = torch.zeros((bs, nq, nc), dtype=torch.float32, device=targets.device)
    # for i, t in enumerate(batch_soft_labels):
    #     src_idx, tgt_idx = match_indices[i]
    #     print("tgt_idx:", tgt_idx)
    #     print("len soft labels:", t.shape[0])
    #     if tgt_idx.shape[0] > 0: # only copy if there are annotations for this image, otherwise keep as all zeros (edge case)
    #         soft_targets[i, tgt_idx, :] = t[src_idx]  # copy into target tensor

    gt_groups = batch["gt_groups"]
    gt_offsets = torch.cumsum(
                    torch.tensor([0] + gt_groups[:-1]),
                    dim=0
                )
    for i, t in enumerate(batch_soft_labels):
        src_idx, tgt_idx = match_indices[i]

        # ensure correct dtype
        src_idx = src_idx.long()
        tgt_idx = tgt_idx.long()

        offset = gt_offsets[i]
        local_tgt_idx = tgt_idx - offset

        # fix empty / malformed tensors
        if t.ndim == 1:
            t = t.reshape(0, nc)

        # skip invalid cases
        if len(src_idx) == 0 or t.shape[0] == 0:
            continue

        # print("tgt_idx:", tgt_idx, "src_idx:", src_idx)
        # print("local_tgt_idx:", local_tgt_idx)
        # # print("len soft labels:", t.shape[0])
        # print(f"[DEBUG] nq={nq}, matches={len(src_idx)}, soft={t.shape[0]}")

        soft_targets[i, src_idx, :] = t[local_tgt_idx]  # copy into target tensor
    
        
    # print("Soft targets shape:", soft_targets.shape)
    # print("Targets shape:", targets.shape)

    # print("GT Scores shape:", gt_scores.shape)

    # ----------------------------
    # Compute loss with soft targets instead of hard one-hot targets
    # ----------------------------

    soft_gt_scores = gt_scores.view(bs, nq, 1) * soft_targets

    # gt scores are IoU values for the nq predictions

    if self.fl:
        if num_gts and self.vfl:
            soft_loss_cls = self.vfl(pred_scores, soft_gt_scores, one_hot)
        else:
            print("Fallback to original loss function since VFL is not enabled.")
            soft_loss_cls = self.fl(pred_scores, one_hot.float())
        soft_loss_cls /= max(num_gts, 1) / nq
    else:
        soft_loss_cls = torch.nn.BCEWithLogitsLoss(reduction="none")(pred_scores, soft_gt_scores).mean(1).sum()

    return {name_class: soft_loss_cls * self.loss_gain["class"]}


loss_module.RTDETRDetectionLoss._get_loss_class = patched_get_loss_class

def align_soft_labels_to_hard_labels(self, soft_labels, hard_labels):
    soft_cls = soft_labels.argmax(dim=1)  # (N_soft,)

    # (N_hard, N_soft)
    match = (hard_labels[:, None] == soft_cls[None, :]) # matching matrix for classes

    # convert to float for argmax trick
    match_float = match.float()

    # cumulative max trick to enforce "first occurrence"
    idxs = []
    last_idx = 0

    for i in range(len(hard_labels)):
        row = match_float[i]

        # mask out already used positions
        row[:last_idx] = 0

        if row.sum() == 0:
            continue  # no match found

        idx = torch.argmax(row)
        idxs.append(idx.item())
        last_idx = idx + 1

    return soft_labels[idxs]

loss_module.RTDETRDetectionLoss.align_soft_labels_to_hard_labels = align_soft_labels_to_hard_labels

OriginalGetLoss = loss_module.RTDETRDetectionLoss._get_loss

def patched_get_loss(
    self,
    pred_bboxes,
    pred_scores,
    gt_bboxes,
    gt_cls,
    gt_groups,
    *args,
    **kwargs,
):
    # 🔥 compute match_indices exactly like original
    match_indices = kwargs.get("match_indices", None)

    if match_indices is None:
        match_indices = self.matcher(
            pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups
        )

    # ✅ STORE THEM HERE
    self._soft_match_indices = match_indices

    # ⚠️ IMPORTANT: pass them forward again
    kwargs["match_indices"] = match_indices

    return OriginalGetLoss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        *args,
        **kwargs,
    )

loss_module.RTDETRDetectionLoss._get_loss = patched_get_loss


OriginalForward = loss_module.RTDETRDetectionLoss.forward

def patched_forward(self, preds, batch, *args, **kwargs):
    # ✅ store full batch BEFORE it gets reduced internally
    self._soft_batch = batch
    # self._dataset_idx = batch.get("dataset_idx", None)
    # self._soft_match_indices = kwargs.get("match_indices", None)
    # print("Patched forward called. Batch keys:", self._soft_batch.keys() if self._soft_batch is not None else "No batch")

    return OriginalForward(self, preds, batch, *args, **kwargs)

loss_module.RTDETRDetectionLoss.forward = patched_forward

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
        "RT-DETR eval on PascalVOC", add_help=True)

    # PascalVOC image directory
    parser.add_argument("--image_dir", type=str,
                        required=True, help="pascalvoc image dir")

    parser.add_argument("--save_dir", type=str, default="predictions/rtdetr-l_soft/pascal/", help="directory to save predictions")

    parser.add_argument("--checkpoint_path", type=str, default="../training/runs/detect/rtdetr-l_pascal_ours_soft/weights/best.pt", help="path to the model checkpoint")


    args = parser.parse_args()

    id_map = { i : i for i in range(20) } # Identity mapping for PascalVOC with 20 classes, mapping only required for COCO

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load fine-tuned RT-DETR model
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint not found at {args.checkpoint_path}. Make sure to train the model first and provide the correct path.")
        sys.exit(1)

    model =  RTDETR(args.checkpoint_path)

    # Replace predictor
    model.predictor = CustomRTDETRPredictor(overrides=model.overrides)
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