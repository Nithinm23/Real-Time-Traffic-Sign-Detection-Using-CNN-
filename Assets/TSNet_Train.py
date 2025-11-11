import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

path = "Data"
labelFile = "labels.csv"
batch_size_val = 50
epochs_val = 50
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

print("Loading image data...")
images = []
classNo = []

if not os.path.isdir(path):
    raise FileNotFoundError(f'Data folder "{path}" not found.')

try:
    myList = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))],
                    key=lambda s: int(s))
except ValueError:
    raise ValueError('Subfolders in "Data" must be named with integers (e.g., 0, 1, 2, ...).')

print(f"Total Classes Detected: {len(myList)}")
noOfClasses = len(myList)

valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
for folder in myList:
    folder_path = os.path.join(path, folder)
    print(f"Loading class {folder}...", end='\r')
    for y in os.listdir(folder_path):
        curImg_path = os.path.join(folder_path, y)
        if os.path.splitext(curImg_path)[1].lower() in valid_exts:
            curImg = cv2.imread(curImg_path)
            if curImg is not None:
                curImg_resized = cv2.resize(curImg, (imageDimensions[1], imageDimensions[0]))
                images.append(curImg_resized)
                classNo.append(int(folder))

images = np.array(images)
classNo = np.array(classNo)

if len(images) == 0:
    raise RuntimeError("No images were loaded. Check your Data folder.")


class_order = [int(x) for x in myList]
with open("class_order.json", "w") as f:
    json.dump(class_order, f)
print("\nClass order mapping saved to class_order.json (compatible with pipeline)")


X_train, X_test, y_train, y_test = train_test_split(
    images, classNo, test_size=testRatio, random_state=42, stratify=classNo
)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=validationRatio, random_state=42, stratify=y_train
)

print("\nData Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)


def preprocessing(img):
    """Applies grayscale, histogram equalization, and normalization."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0  
    return img

X_train = np.array([preprocessing(img) for img in X_train])
X_validation = np.array([preprocessing(img) for img in X_validation])
X_test = np.array([preprocessing(img) for img in X_test])


X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]
X_test = X_test[..., np.newaxis]

dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)

y_train = to_categorical(y_train, num_classes=noOfClasses)
y_validation = to_categorical(y_validation, num_classes=noOfClasses)
y_test = to_categorical(y_test, num_classes=noOfClasses)


y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights_dict = dict(enumerate(class_weights))
print("\nCalculated Class Weights for training.")

def create_improved_model():
    """Defines and compiles the improved CNN model."""
    model = Sequential()

    # --- Block 1 ---
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=(imageDimensions[0], imageDimensions[1], 1)))
    model.add(BatchNormalization()) # ADDED: Stabilizes training
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # --- Block 2 ---
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax')) 

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_improved_model()
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

print("\nStarting model training...")
history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    class_weight=class_weights_dict,
    callbacks=[early_stop, learning_rate_reduction], 
    verbose=1
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

print("\nEvaluating on Test Data...")
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {score[0]:.4f}')
print(f'Test Accuracy: {score[1]:.4f}')


model.save("CNN.keras")
print("\nImproved model saved as model_trained.keras")
