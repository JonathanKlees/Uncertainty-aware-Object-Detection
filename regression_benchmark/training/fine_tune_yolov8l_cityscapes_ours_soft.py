#
#
# NOTE: This file contains modified source code of the ultralytics library,
# Much content is just copied from the repository and slightly modified,
# check original source for licensing information:
# https://github.com/ultralytics/ultralytics
#
#

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel
import torch.nn.functional as F
import torch
import torch.nn as nn

from typing import Any
from ultralytics.utils.tal import make_anchors, TaskAlignedAssigner
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils import LOGGER

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
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            # extract only soft label parts
            soft = [list(map(float, line.strip().split()[5:])) for line in lines]

            return torch.tensor(soft, dtype=torch.float32)

        except FileNotFoundError: # For the case of images with zero annotations
            return torch.zeros((0, 8), dtype=torch.float32)  # return all zeros if no annotations

    def collate_fn(self, batch):
        out = super().collate_fn(batch)

        if "soft_cls" in batch[0]:
            out["soft_cls"] = [b["soft_cls"] for b in batch]

        return out

class SoftTaskAlignedAssigner(TaskAlignedAssigner):
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, soft_gt_labels=None):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # print(">>> USING SOFT ASSIGNER <<<")
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, nc)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # replicate target scores for soft labels if available
        if soft_gt_labels is not None:
            # init all zero array of (b, h*w, nc) and fill in soft labels for foreground points according to target_gt_idx
            soft_target_scores = torch.zeros(
                (target_labels.shape[0], target_labels.shape[1], self.num_classes),
                dtype=torch.float32,
                device=target_labels.device,
            )  
            soft_target_scores = soft_gt_labels.view(-1, self.num_classes)[target_gt_idx]  # (b, h*w, num_classes)
            soft_target_scores = torch.where(fg_scores_mask > 0, soft_target_scores, 0)  # mask only keeps foreground points
        else:
            print("Warning: No soft labels provided to assigner, falling back to hard labels for scores.")
            soft_target_scores = target_scores.clone()  # fallback to hard labels if no soft labels available

        return target_labels, target_bboxes, target_scores, soft_target_scores
    
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, soft_gt_labels=None):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            soft_gt_labels (torch.Tensor): Soft ground truth labels with shape (bs, n_max_boxes, num_classes). (ADDED)
        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
            soft_target_scores (torch.Tensor): Soft Target scores with shape (bs, num_total_anchors, num_classes). (ADDED)

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, soft_gt_labels)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Move tensors to CPU, compute, then move back to original device
                LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
                cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, soft_gt_labels)]
                result = self._forward(*cpu_tensors)
                return tuple(t.to(device) for t in result)
            raise

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, soft_gt_labels):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            soft_gt_labels (torch.Tensor): Soft ground truth labels with shape (bs, n_max_boxes, num_classes) (ADDED).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
            soft_target_scores (torch.Tensor): Soft Target scores with shape (bs, num_total_anchors, num_classes) (ADDED).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        # Assigned target
        target_labels, target_bboxes, target_scores, soft_target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask, soft_gt_labels)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        soft_target_scores = soft_target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx, soft_target_scores


class SoftDetectionModel(DetectionModel):

    def init_criterion(self):
        return SoftDetectionLoss(self)
    
class SoftDetectionLoss(v8DetectionLoss):

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = SoftTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)


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

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Load soft labels and align them to hard labels which are sometimes already filtered.

        batch_soft_labels = []
        for i in range(batch_size): # align per image
            hard_labels = batch["cls"][batch["batch_idx"] == i].squeeze(1)  # (N_hard,) hard labels for this image
            soft_labels = batch["soft_cls"][i].to(targets.device)  # (N_soft, num_classes)

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

            # # convert to int (hard labels) for sanity check -> reproduces hard label results
            # soft_labels = soft_labels.to(torch.int64)

            # soft_labels =soft_labels.to(hard_labels.dtype) # convert to same dtype as hard labels for loss calculation

            batch_soft_labels.append(soft_labels)

        # apply same padding to soft labels as hard labels for batch processing
        max_num_obj = max(t.shape[0] for t in batch_soft_labels)
        n_c = self.nc  # model's number of classes

        # initialize soft_gt_labels with zeros of expected shape (batch_size, max_num_obj, num_classes) and fill in values from batch_soft_labels
        soft_gt_labels = torch.zeros(len(batch_soft_labels), max_num_obj, n_c, dtype=batch_soft_labels[0].dtype, device=batch_soft_labels[0].device)
        for i, t in enumerate(batch_soft_labels):
            if t.shape[0] > 0: # only copy if there are annotations for this image, otherwise keep as all zeros (edge case)
                soft_gt_labels[i, :t.shape[0]] = t  # copy into padded tensor

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx, soft_target_scores = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            soft_gt_labels,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        soft_target_scores_sum = max(soft_target_scores.sum(), 1)
        
        # Updated classification loss with soft labels
        if "soft_cls" in batch:

            orig_bce_loss = self.bce(pred_scores, target_scores.to(dtype))
            soft_bce_loss = self.bce(pred_scores, soft_target_scores.to(dtype))  # (bs, num_anchors, nc)
            # kl_loss = F.kl_div(F.log_softmax(pred_scores, dim=-1), soft_target_scores, reduction='none')  # KL divergence loss for soft labels

            # print("Original BCE Loss:", orig_bce_loss.sum())
            # print("Soft BCE Loss:", soft_bce_loss.sum())

           # loss[1] = orig_bce_loss.sum() / target_scores_sum 
            loss[1] = soft_bce_loss.sum() / soft_target_scores_sum # use soft labels for classification loss

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
                # target_scores,
                soft_target_scores,  # use soft labels for bbox loss weighting as well
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




if __name__ == "__main__":

    model = YOLO("../inference/checkpoints/yolov8l.pt")

    model.train(
        data="configs/cityscapes_ours.yaml",
        epochs=100,
        trainer=SoftYOLOTrainer,
        optimizer = "SGD",
        lr0 = 1e-4,
        name = "yolov8l_cityscapes_ours_soft_bce_only",
        mosaic=False,  # disable mosaic augmentation for better debugging and analysis of soft labels
    )