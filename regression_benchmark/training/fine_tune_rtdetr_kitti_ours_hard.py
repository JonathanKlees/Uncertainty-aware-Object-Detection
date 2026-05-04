from ultralytics import RTDETR

model = RTDETR("../inference/checkpoints/rtdetr-l.pt") # already downloaded weights under inference

results = model.train(data="configs/kitti_ours.yaml", epochs=100, mosaic = False, name="rtdetr-l_kitti_ours_hard")