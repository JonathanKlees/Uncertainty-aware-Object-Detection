import argparse
import os
import sys
import time
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

# from torchvision.datasets import CocoDetection
import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list

        # import ipdb; ipdb.set_trace()

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # filt invalid boxes/masks/keypoints
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target

class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, cat_list=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert cat_list is not None
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)  # 80, 256. normed

        id_map = {i : i for i in range(7)} # Identity mapping for KITTI 7 classes.
        # build a mapping from label_id to pos_map
        new_pos_map = torch.zeros((7, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = self.positive_map.to(prob_to_token.device)
        # (bs, 100, 256) @ (7, 256).T -> (bs, 100, 7)
        prob_to_label = prob_to_token @ pos_maps.T

        # logits_per_class = out_logits @ pos_maps.T # added: aggregate logits over tokens for each class -> Causes NaN due to -inf values in out_logits.

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Added: Gather top-k class probability distributions
        probs_of_selected_queries = torch.gather(
            prob,
            1,
            topk_boxes.unsqueeze(-1).expand(-1, -1, prob.shape[2])
        )

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b, "probs": p}
                   for s, l, b, p in zip(scores, labels, boxes, probs_of_selected_queries)] # added logits per class to results

        return results

def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()

    # build coco style dataset and dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    anno_path = os.path.dirname(args.label_dir) + "/instances_val.json" # COCO style annotation file -> is created if not existing

    cat_list=[ 
            "car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram"
        ]

    if not os.path.exists(anno_path): # if no coco style annotation file exists
        print("No COCO style annotation file found. Please create one using dataset_tools/convert_yolo_to_coco.ipynb ...")


    dataset = CocoDetection(
        args.image_dir, anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build captions
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        cat_list=cat_list, tokenlizer=tokenlizer)


    # run inference
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):

        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # feed to the model
        outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)

        # --- SAVE per-image predictions as JSON ---
        for target, res in zip(targets, results):
            image_id = target["image_id"]
            save_dict = {
                "image_id": image_id,
                "boxes": res["boxes"].cpu().tolist(),   # convert to list
                "scores": res["scores"].cpu().tolist(),
                "labels": res["labels"].cpu().tolist(),
                "probs": res["probs"].cpu().tolist(), # class probabilities per detection
            }
            save_path = os.path.join(args.save_dir, f"{image_id}.json")
            with open(save_path, "w") as f:
                json.dump(save_dict, f, indent=2)

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i+1}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on KITTI", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # dataset info
    parser.add_argument("--image_dir", type=str,
                        required=True, help="dataset image dir")
    parser.add_argument("--label_dir", type=str,
                        required=True, help="dataset label dir")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    
    parser.add_argument("--save_dir", type=str, default="predictions/gdino/kitti/", help="directory to save predictions")
    args = parser.parse_args()

    if os.path.exists(args.save_dir):
        os.system(f"rm -rf {args.save_dir}") # remove potentially existing predictions (we changed filenames for storing, so removal is a safeguard)
    os.makedirs(args.save_dir) # create new directory for predictions

    main(args)