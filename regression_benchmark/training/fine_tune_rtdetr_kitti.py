from ultralytics import RTDETR

model = RTDETR("../inference/checkpoints/rtdetr-l.pt") # already downloaded weights under inference

results = model.train(data="configs/kitti.yaml", epochs=100, optimizer = "SGD", lr0 = 1e-4, name="rtdetr-l_kitti")