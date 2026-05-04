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

CLASS_MAP = { i: i+1 for i in range(20) } # 0-19 mapped to 1-20 because Faster R-CNN expects background class at 0

class PascalVOCDataset(Dataset):
    def __init__(self, img_dir, ann_file, soft_label_threshold=0.5):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.soft_dataset = json.load(open(self.ann_file, "r"))["objects"] # load soft labels from JSON
        self.soft_label_threshold = soft_label_threshold
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        W = img.width
        H = img.height

        # using soft labels for this dataloader

        boxes = []
        labels = []

        if img_name in self.soft_dataset: # some images do not have annotations

            for obj in self.soft_dataset[img_name]: # get annotations for this image (without .jpg)
                
                if max(obj["soft_label"][:-1]) < self.soft_label_threshold:
                    continue # skip low-confidence annotations (excluding last element which is "cant solve")

                cls = int(np.argmax(obj["soft_label"][:-1])) # get class with highest soft label (excluding "cant solve")

                boxes.append(obj["bbox"]) # [x_min, y_min, x_max, y_max]
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
    train_img_dir = "/path/to/datasets/VOC/images/train2012"
    train_ann_dir = "../../our_datasets/soft_PascalVOC_2012_detection_train.json"

    results_dir = "runs/frcnn_pascal_ours_hard" # to store results
    os.makedirs(results_dir, exist_ok=True)

    # create dataset and dataloader
    pascal_train = PascalVOCDataset(train_img_dir, train_ann_dir)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        pascal_train,
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
            torch.save(model.state_dict(), f"{results_dir}/frcnn_pascal_ours_hard_best.pth")

        # step scheduler AFTER epoch
        scheduler.step()

    # save final model
    torch.save(model.state_dict(), f"{results_dir}/frcnn_pascal_ours_hard_last.pth")

    # save log to JSON
    with open(f"{results_dir}/frcnn_pascal_ours_hard_log.json", "w") as f:
        json.dump(log, f, indent=4)