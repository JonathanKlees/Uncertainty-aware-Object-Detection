from ultralytics import settings

settings.update({"datasets_dir": "/path/to/datasets"}) # set dataset dir

from ultralytics import settings # reload settings from disk

from ultralytics import YOLO

model = YOLO("../inference/checkpoints/yolov8l.pt") # already downloaded weights under inference

results = model.train(data="configs/VOC_ours.yaml",
                            epochs=100, 
                            optimizer = "SGD",
                            lr0 = 1e-4,
                            mosaic=False,  # disable mosaic augmentation for better debugging and analysis of soft labels
                            name="yolov8l_pascal_ours_hard")