import cv2
import numpy as np
from ultralytics import YOLO

coco_model = YOLO("yolo11s-seg.pt")
coco_classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

custom_model = YOLO("runs/detect/train2/weights/best.pt")
custom_classNames = ["traffic sign"]

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(coco_classNames), 3), dtype=np.uint8)


img = cv2.imread("test_2.jpg")
output_img = img.copy()


print("Processing general object detection (COCO Model)...")
coco_results = coco_model(img, conf=0.3)

for r in coco_results:
    boxes = r.boxes
    masks = r.masks

    if masks is not None:
        for i, mask_data in enumerate(masks.data):
            mask = mask_data.cpu().numpy()

            cls_id = int(boxes[i].cls[0])
            color = [int(c) for c in colors[cls_id]]

            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = mask_resized > 0.5

            mask_img = np.zeros_like(img, dtype=np.uint8)
            mask_img[mask_binary] = color

            output_img = cv2.addWeighted(output_img, 1, mask_img, 0.5, 0)

            label = coco_classNames[cls_id]
            x1, y1, _, _ = map(int, boxes[i].xyxy[0])
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


print("Processing traffic sign detection (Custom Model)...")

custom_results = custom_model(img, conf=0.2, iou=0.2)


for r in custom_results:
    for box in r.boxes:
        cls_id = int(box.cls[0])  
        label = custom_classNames[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cropped_sign = img[y1:y2, x1:x2]

        cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 255), 2)  
        cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


print("Processing complete. Displaying final image.")
frame = cv2.resize(output_img, (720, 420))

cv2.imshow("Combined Detections", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
