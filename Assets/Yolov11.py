import cv2
import math
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")

classNames = [
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

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype=np.uint8)


img = cv2.imread("122_jpg.rf.8cb866b26d53be28b873b34154a4f7f2.jpg")

results = model(img)


output_img = img.copy()

for r in results:
    masks = r.masks
    boxes = r.boxes

    if masks is not None:
        mask_array = masks.data.cpu().numpy()

        for i, mask in enumerate(mask_array):
            cls_id = int(boxes[i].cls[0])
            color = [int(c) for c in colors[cls_id]]

            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = mask_resized > 0.5

            mask_img = np.zeros_like(img, dtype=np.uint8)
            mask_img[mask_binary] = color

            output_img = cv2.addWeighted(output_img, 1, mask_img, 0.5, 0)

            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
            label = classNames[cls_id] if cls_id < len(classNames) else f"class{cls_id}"

            cv2.putText(output_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

output_img = cv2.resize(output_img,(730,480))

cv2.imshow("YOLOv11 Segmentation and Object detection ", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
