import argparse
import os
from tqdm import tqdm
import json
import torch
import torch.multiprocessing as mp
import numpy as np
import shutil
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Adjust Post-Processing in Faster R-CNN to output full class probabilities instead of just the max class score

from typing import Optional
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference, keypointrcnn_loss, keypointrcnn_inference
from torchvision.ops import boxes as box_ops

class CustomRoIHeads(RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )

    def postprocess_detections(
        self, class_logits, box_regression, proposals, image_shapes
    ):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(p) for p in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # class probabilities (including background)
        pred_probs = F.softmax(class_logits, dim=-1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_probs_list = pred_probs.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_probs = []

        for boxes, probs, image_shape in zip(pred_boxes_list, pred_probs_list, image_shapes):

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # remove background class for ranking
            scores = probs[:, 1:]           # exclude background
            labels = torch.arange(1, num_classes, device=device)

            labels = labels.view(1, -1).expand_as(scores)

            # remove background
            boxes = boxes[:, 1:, :]      # [N, C-1, 4]
            scores = probs[:, 1:]        # [N, C-1]

            labels = torch.arange(1, num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # flatten everything
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # thresholding
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # NMS
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # store full probability vector for each selected detection
            # map back to original proposal index
            proposal_indices = keep // (num_classes - 1)
            full_probs = probs[proposal_indices]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_probs.append(full_probs)

        return all_boxes, all_scores, all_labels, all_probs
    
    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels, probs = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "probs": probs[i], # updated only this part here (and 8 lines above) to include full class probabilities
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Faster R-CNN eval on PascalVOC", add_help=True)

    # PascalVOC image directory
    parser.add_argument("--image_dir", type=str,
                        required=True, help="pascal image dir")
    
    parser.add_argument("--model_name", type=str, default="frcnn", help="name of the model")

    parser.add_argument("--checkpoint_path", type=str, default="../training/runs/frcnn_pascal/frcnn_pascal_best.pth", help="path to model checkpoint")

    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="confidence threshold for predictions")
    parser.add_argument("--max_detections", type=int, default=300, help="maximum number of detections per image")
    parser.add_argument("--save_dir", type=str, default="predictions/frcnn/pascal/", help="directory to save predictions")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint_path}. Fine-tune the model first and make sure the checkpoint path is correct.")

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load pre-trained Faster R-CNN model 
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, # we will load our own checkpoint, so we set weights to None
        box_score_thresh=args.confidence_threshold
    )
    model.roi_heads.detections_per_img = args.max_detections # we limit to 300 detections per image as in the original Faster R-CNN paper
    # this makes additional confidence thresholding obsolete, so we set it to 0 by default

    # Replace roi_heads with slightly different custom version
    roi_heads = model.roi_heads

    model.roi_heads = CustomRoIHeads(
        roi_heads.box_roi_pool,
        roi_heads.box_head,
        roi_heads.box_predictor,
        roi_heads.proposal_matcher.high_threshold,
        roi_heads.proposal_matcher.low_threshold,
        roi_heads.fg_bg_sampler.batch_size_per_image,
        roi_heads.fg_bg_sampler.positive_fraction,
        roi_heads.box_coder.weights,
        roi_heads.score_thresh,
        roi_heads.nms_thresh,
        roi_heads.detections_per_img,
    )

    # Load our trained checkpoint
    # -> adjust classification head to match our number of classes (20 Pascal classes + background)
    num_classes = 20 + 1 # 20 Pascal classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(args.checkpoint_path)) # load checkpoint with our trained weights

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inference on PascalVOC
    for img in tqdm(os.listdir(args.image_dir), desc="Running inference on PascalVOC images"):
        image = Image.open(os.path.join(args.image_dir, img)).convert("RGB")
        inputs = torch.unsqueeze(torch.from_numpy(np.array(image)).permute(2, 0, 1)/255.0, dim=0).to(device) # convert to tensor and add batch dimension
        with torch.no_grad():
            results = model(inputs)

        r = results[0] # batch size is 1, so we take the first element

        bg_probs = r["probs"][:, 0] # background probabilities
        fg_probs = r["probs"][:, 1:] # foreground class probabilities

        # add background prob as last entry
        probs = torch.cat([fg_probs, bg_probs.unsqueeze(1)], dim=1)  # [num_detections, num_classes + 1] with background as last entry

        image_id = int(os.path.basename(img).split(".")[0])
        save_dict = {
            "image_id": image_id,
            "boxes": r["boxes"].cpu().numpy().tolist(),
            "labels": r["labels"].cpu().numpy().tolist(), # already PascalVOC category IDs (20 classes)
            "scores": r["scores"].cpu().numpy().tolist(),
            "probs": probs.cpu().numpy().tolist() # full class probabilities
        }

        # Store the predictions in COCO format (one json per image)
        with open(os.path.join(save_dir, f"{image_id}.json"), "w") as f: 
            json.dump(save_dict, f, indent = 2)