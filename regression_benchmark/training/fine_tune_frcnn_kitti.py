# fine tune Faster R-CNN on custom dataset
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# CLASS_MAP = {
#     "car": 1,
#     "van": 2,
#     "truck": 3,
#     "pedestrian": 4,
#     "person_sitting": 5,
#     "cyclist": 6,
#     "tram": 7
# }
CLASS_MAP = { i: i+1 for i in range(7) } # 0-6 mapped to 1-7 because Faster R-CNN expects background class at 0

class KittiDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        W = img.width
        H = img.height
        # using yolo labels for this dataloader

        with open(os.path.join(self.ann_dir, img_name.replace(".png", ".txt"))) as f:
            lines = f.readlines()
        
        boxes = []
        labels = []

        for line in lines:
            cls, xc, yc, w, h = map(float, line.strip().split())

            cls = int(cls)

            # convert YOLO → xyxy
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_MAP[cls]) # ⚠️ shift by +1 for Faster R-CNN

        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)


        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return torch.tensor(np.array(img)).permute(2, 0, 1) / 255.0, target
    
if __name__ == "__main__":

    # set dataset paths
    train_img_dir = "/path/to/datasets/KITTI/train/images"
    train_ann_dir = "/path/to/datasets/KITTI/train/labels"

    results_dir = "runs/frcnn_kitti" # to store results
    os.makedirs(results_dir, exist_ok=True)

    # create dataset and dataloader
    kitti_train = KittiDataset(train_img_dir, train_ann_dir)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        kitti_train,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    num_classes = len(CLASS_MAP) + 1 # +1 for background

    # load pretrained Faster R-CNN
    model = fasterrcnn_resnet50_fpn_v2(weights="FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1")

    # replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # training parameters
    epochs = 50
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # LR scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5) # halve LR every 10 epochs

    # logging
    log = []
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix({"loss": loss.item()})

        # average loss over epoch
        epoch_loss /= len(train_loader)

        # get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: Avg Loss = {epoch_loss:.4f}, LR = {current_lr:.6f}")

        # log metrics
        log.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "lr": current_lr
        })

        # save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"{results_dir}/frcnn_kitti_best.pth")

        # step scheduler AFTER epoch
        scheduler.step()

    # save final model
    torch.save(model.state_dict(), f"{results_dir}/frcnn_kitti_last.pth")

    # save log to JSON
    with open(f"{results_dir}/frcnn_kitti_log.json", "w") as f:
        json.dump(log, f, indent=4)