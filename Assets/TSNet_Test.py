import numpy as np
import cv2
from tensorflow import keras
import json
import pandas as pd

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.80  
font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)


model = keras.models.load_model("model_trained.keras")


with open("class_order.json") as f:
    idx_to_true_class_id = json.load(f)

labels_df = pd.read_csv("labels.csv")  
id_to_name = dict(zip(labels_df["ClassId"], labels_df["Name"]))


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img


while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)

    cv2.imshow("Processed Image", (img * 255).astype(np.uint8))

    img_input = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img_input, verbose=0)
    classIndex = int(np.argmax(predictions))             
    true_class_id = idx_to_true_class_id[classIndex]     
    probabilityValue = float(np.max(predictions))

    cv2.putText(imgOriginal, "CLASS:", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY:", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    if probabilityValue > threshold:
        className = id_to_name.get(true_class_id, f"ID {true_class_id}")
        cv2.putText(imgOriginal, f"{true_class_id} - {className}", (120, 35),
                    font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (220, 75),
                    font, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
  
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
