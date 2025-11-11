from ultralytics import YOLO
model = YOLO("yolo11s.yaml")
model.load("yolo11s.pt")
results = model.train(data="Yolo.yaml", epochs=15,)
