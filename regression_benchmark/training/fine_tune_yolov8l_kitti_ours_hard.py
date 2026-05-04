from ultralytics import YOLO

model = YOLO("../inference/checkpoints/yolov8l.pt") # already downloaded weights under inference

results = model.train(data="configs/kitti_ours.yaml",
                            epochs=100, 
                            mosaic=False,  # disable mosaic augmentation for better debugging and analysis of soft labels
                            optimizer = "SGD",
                            lr0 = 1e-4,
                            name="yolov8l_kitti_ours_hard")