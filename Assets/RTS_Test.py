import cv2
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO
from tensorflow import keras
import pyttsx3
import re
import time
import psutil
import os

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    return round(mem, 2)

def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

print("Loading models and label files...")

TSNet = pyttsx3.init()

coco_model = YOLO("yolo11s-seg.pt")
custom_model = YOLO("runs/detect/train2/weights/best.pt")

cnn_model = keras.models.load_model("model_trained.keras")

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
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(coco_classNames), 3), dtype=np.uint8)

custom_classNames = ["traffic sign"]

with open("class_order.json") as f:
    class_order_list = json.load(f)
    idx_to_true_class_id = {i: v for i, v in enumerate(class_order_list)}

labels_df = pd.read_csv("labels.csv")
id_to_name = dict(zip(labels_df["ClassId"], labels_df["Name"]))

CNN_PROBABILITY_THRESHOLD = 0.80

def preprocess_for_cnn(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

img = cv2.imread("122_jpg.rf.8cb866b26d53be28b873b34154a4f7f2.jpg")
output_img = img.copy()

total_start = time.time()
start_mem_total = get_memory_usage_mb()

print("Processing general object detection (COCO Model)...")
start_time = time.time()
start_mem = get_memory_usage_mb()

coco_results = coco_model(img, conf=0.3)

end_time = time.time()
end_mem = get_memory_usage_mb()
print(f"COCO detection time: {end_time - start_time:.2f} seconds")
print(f"Memory used during COCO detection: {end_mem - start_mem:.2f} MB")

for r in coco_results:
    if r.masks:
        for i, mask_data in enumerate(r.masks.data):
            mask = mask_data.cpu().numpy()
            cls_id = int(r.boxes[i].cls[0])
            label = coco_classNames[cls_id]

            if label in ["stop sign", "traffic light"]:
                continue

            color = [int(c) for c in colors[cls_id]]
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = mask_resized > 0.5

            mask_img = np.zeros_like(img, dtype=np.uint8)
            mask_img[mask_binary] = color

            output_img = cv2.addWeighted(output_img, 1, mask_img, 0.5, 0)

            x1, y1, _, _ = map(int, r.boxes[i].xyxy[0])
            cv2.putText(output_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


print("Processing traffic sign detection and classification...")
start_time = time.time()
start_mem = get_memory_usage_mb()

custom_results = custom_model(img, conf=0.3, iou=0.3)

end_time = time.time()
end_mem = get_memory_usage_mb()
print(f"Custom YOLO detection time: {end_time - start_time:.2f} seconds")
print(f"Memory used during Custom YOLO: {end_mem - start_mem:.2f} MB")

detected_signs_messages = []
unique_detected_ids = set()
cnn_total_time = 0
cnn_count = 0
start_mem_cnn = get_memory_usage_mb()

for r in custom_results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_sign = img[y1:y2, x1:x2]

        if cropped_sign.size == 0:
            continue

        processed_crop = preprocess_for_cnn(cropped_sign)
        img_input = processed_crop.reshape(1, 32, 32, 1)

        t0 = time.time()
        predictions = cnn_model.predict(img_input, verbose=0)
        t1 = time.time()
        cnn_total_time += (t1 - t0)
        cnn_count += 1

        probabilityValue = float(np.max(predictions))
        final_label = "Unknown Sign"

        if probabilityValue > CNN_PROBABILITY_THRESHOLD:
            model_index = int(np.argmax(predictions))
            if model_index in idx_to_true_class_id:
                true_class_id = idx_to_true_class_id[model_index]
                className = id_to_name.get(true_class_id, f"ID {true_class_id}")

                if true_class_id not in unique_detected_ids:
                    unique_detected_ids.add(true_class_id)
                    speech_message = className

                    if 0 <= true_class_id <= 8:
                        match = re.search(r'\((\d+km/h)\)', className)
                        speed_limit = match.group(1).replace('km/h', ' kilometers per hour') if match else "unknown speed"

                        if true_class_id == 6:
                            speech_message = "End of speed limit 80 kilometers per hour. Proceed with caution."
                        elif true_class_id == 8:
                            speech_message = "Warning! Speed limit is 120 kilometers per hour. Drive carefully."
                        else:
                            speech_message = f"The current speed limit is {speed_limit}. Please adjust your driving speed."

                    elif true_class_id == 13:
                        speech_message = "Caution: Yield sign ahead! Prepare to slow down and give way."

                    elif true_class_id == 14:
                        speech_message = "Emergency! Stop sign detected. You must stop completely."

                    elif 10 <= true_class_id <= 17 or 41 <= true_class_id <= 43:
                        speech_message = f"Important Restriction: {className}. Follow the traffic regulation."

                    elif 18 <= true_class_id <= 31:
                        speech_message = f"Hazard Warning: {className}. Please be extra careful."

                    detected_signs_messages.append(speech_message)

                final_label = f"{className} ({round(probabilityValue * 100, 1)}%)"

        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, final_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

end_mem_cnn = get_memory_usage_mb()
if cnn_count > 0:
    print(f"Average CNN classification time per sign: {cnn_total_time / cnn_count:.3f} seconds")
print(f"Memory used during CNN classification: {end_mem_cnn - start_mem_cnn:.2f} MB")


if detected_signs_messages:
    full_speech = "Attention. " + ". ".join(detected_signs_messages) + "."
    TSNet.say(full_speech)

total_end = time.time()
end_mem_total = get_memory_usage_mb()

print("\n===== SYSTEM PERFORMANCE SUMMARY =====")
print(f"Total pipeline execution time: {total_end - total_start:.2f} seconds")
print(f"Total memory change: {end_mem_total - start_mem_total:.2f} MB")
print(f"CPU Usage: {get_cpu_usage()}%")
print(f"Current RAM Usage: {get_memory_usage_mb()} MB")

print("\nProcessing complete. Displaying final image.")
frame = cv2.resize(output_img, (720, 400))
cv2.imshow("Complete Traffic Sign Pipeline", frame)
TSNet.runAndWait()
cv2.waitKey(0)
cv2.destroyAllWindows()


