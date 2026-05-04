from ultralytics import settings

settings.update({"datasets_dir": "/path/to/datasets"}) # set dataset dir

from ultralytics import settings  # reload settings from disk

from ultralytics import RTDETR

model = RTDETR("../inference/checkpoints/rtdetr-l.pt") # already downloaded weights under inference

results = model.train(data="configs/VOC_ours.yaml", epochs=100, optimizer = "SGD", mosaic = False,
                        lr0 = 1e-4, name = "rtdetr-l_pascal_ours_hard")