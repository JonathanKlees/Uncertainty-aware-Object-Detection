import argparse
import os
from tqdm import tqdm
import json
import torch
import torch.multiprocessing as mp
import numpy as np
import shutil
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image

# Patch post-processing to return full class probabilities instead of just the max class
class CustomOwlViTProcessor(OwlViTProcessor):
    def post_process_object_detection(
            self,
            outputs: "OwlViTObjectDetectionOutput",
            threshold: float = 0.1,
            target_sizes: None = None,
        ):
            """
            Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
            bottom_right_x, bottom_right_y) format.

            Args:
                outputs ([`OwlViTObjectDetectionOutput`]):
                    Raw outputs of the model.
                threshold (`float`, *optional*, defaults to 0.1):
                    Score threshold to keep object detection predictions.
                target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                    Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                    `(height, width)` of each image in the batch. If unset, predictions will not be resized.

            Returns:
                `list[Dict]`: A list of dictionaries, each dictionary containing the following keys:
                - "scores": The confidence scores for each predicted box on the image.
                - "labels": Indexes of the classes predicted by the model on the image.
                - "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
            """
            batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
            batch_size = len(batch_logits)

            if target_sizes is not None and len(target_sizes) != batch_size:
                raise ValueError("Make sure that you pass in as many target sizes as images")

            # batch_logits of shape (batch_size, num_queries, num_classes)
            batch_probs = torch.sigmoid(batch_logits)  # (B, Q, C)

            # Convert to [x0, y0, x1, y1] format
            batch_boxes = center_to_corners_format(batch_boxes)

            # Convert from relative [0, 1] to absolute [0, height] coordinates
            if target_sizes is not None:
                batch_boxes = _scale_boxes(batch_boxes, target_sizes)

            results = []
            for probs, boxes in zip(batch_probs, batch_boxes):

                scores, labels = torch.max(probs, dim=-1)

                keep = scores > threshold

                results.append({
                    "scores": scores[keep],
                    "labels": labels[keep],
                    "boxes": boxes[keep],
                    "probs": probs[keep]   # full class distribution
                })

            return results
    
# Utils copied from Hugging Face:

def _center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


def _center_to_corners_format_numpy(bboxes_center: np.ndarray) -> np.ndarray:
    center_x, center_y, width, height = bboxes_center.T
    bboxes_corners = np.stack(
        # top left x, top left y, bottom right x, bottom right y
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1,
    )
    return bboxes_corners

def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes

# function below inspired by https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def center_to_corners_format(bboxes_center):
    """
    Converts bounding boxes from center format to corners format.

    center format: contains the coordinate for the center of the box and its width, height dimensions
        (center_x, center_y, width, height)
    corners format: contains the coordinates for the top-left and bottom-right corners of the box
        (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    # Function is used during model forward pass, so we use torch if relevant, without converting to numpy
    if isinstance(bboxes_center, torch.Tensor):
        return _center_to_corners_format_torch(bboxes_center)
    elif isinstance(bboxes_center, np.ndarray):
        return _center_to_corners_format_numpy(bboxes_center)

    raise ValueError(f"Unsupported input type {type(bboxes_center)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Owl-ViT eval on Cityscapes", add_help=True)

    # Cityscapes image directory
    parser.add_argument("--image_dir", type=str,
                        required=True, help="cityscapes image dir")
    
    parser.add_argument("--model_name", type=str, default="owlvit", help="name of the model")

    parser.add_argument("--confidence_threshold", type=float, default=0.001, help="confidence threshold for predictions")

    parser.add_argument("--save_dir", type=str, default="predictions/owlvit/cityscapes/", help="directory to save predictions")

    args = parser.parse_args()

    id_map = {i : i for i in range(8)} # Identity mapping for the 8 classes in Cityscapes, map is only required for COCO

    classes = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    text_labels = [[f"a photo of a {c}" for c in classes]] # prompts

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load pre-trained Owl-ViT model with custom post-processing
    processor = CustomOwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Inference on Cityscapes
    for img in tqdm(os.listdir(args.image_dir), desc="Running inference on Cityscapes images"):
        image = Image.open(os.path.join(args.image_dir, img)).convert("RGB")
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(outputs=outputs, threshold=args.confidence_threshold, target_sizes=[image.size[::-1]])

        r = results[0] # batch size is 1, so we take the first element

        image_id = os.path.basename(img).split(".")[0]
        save_dict = {
            "image_id": image_id,
            "boxes": r["boxes"].cpu().numpy().tolist(),
            "labels": [id_map[int(l)] for l in r["labels"].cpu().numpy().tolist()],
            "scores": r["scores"].cpu().numpy().tolist(),
            "probs": r["probs"].cpu().numpy().tolist() # full class probabilities
        }

        # Store the predictions in COCO format (one json per image)
        with open(os.path.join(save_dir, f"{image_id}.json"), "w") as f: 
            json.dump(save_dict, f, indent = 2)