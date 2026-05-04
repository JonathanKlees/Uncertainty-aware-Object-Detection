from ultralytics import settings

settings.update({"datasets_dir": "/path/to/datasets"}) # set dataset dir

from ultralytics import settings # reload settings from disk

from ultralytics import RTDETR

model = RTDETR("../inference/checkpoints/rtdetr-l.pt") # already downloaded weights under inference

results = model.train(data="configs/cityscapes_ours.yaml", epochs=100, mosaic = False, name="rtdetr-l_cityscapes_ours_hard")