from ultralytics import settings

settings.update({"datasets_dir": "/path/to/datasets"}) # set dataset dir

from ultralytics import settings # reload settings from disk

from ultralytics import YOLO

model = YOLO("../inference/checkpoints/yolov8l.pt") # already downloaded weights under inference



results = model.train(data="configs/COCO_ours.yaml",
                            epochs=100, 
                            warmup_epochs = 0.5,
                            optimizer = "SGD",
                            lr0 = 1e-5, # lower learning rate for COCO fine-tuning
                            freeze = 10, # freeze backbone to stabilize fine-tuning 
                            mosaic=False,  # disable mosaic augmentation for better debugging and analysis of soft labels
                            name="yolov8l_coco_ours_hard")