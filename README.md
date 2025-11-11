# Project RTS - Real-Time Traffic Sign Detection Using CNN

# Overview
<p align="justify">
Project RTS is an advanced, real-time traffic sign detection and recognition system designed to enhance road safety and support intelligent transportation systems. Leveraging Deep Learning and Computer Vision, the system identifies and classifies traffic signs from live video feeds, providing instant feedback crucial for Advanced Driver Assistance Systems (ADAS) and autonomous vehicles.
The model is trained on diverse datasets of Indian and German traffic signs to ensure robust performance across varying lighting, distance, and weather conditions.

## Abstract
<p align="justify">
Project RTS implements a two-stage real-time detection and classification pipeline.  
The system first employs YOLOv11s-seg for traffic sign detection and localization, then passes cropped images to a lightweight CNN classifier (TSNet) for classification.  
Trained on a dataset of 42,235 labeled images across 47 traffic sign categories, TSNet delivers exceptional precision.  
The integrated model operates in real time on standard computing hardware without GPU acceleration and includes pyttsx3 voice feedback for driver alerts.  

# Key Achievements
- Real-time detection and recognition (~30 FPS)  
- Overall accuracy: 98.1% 
- Supports 47 distinct traffic sign classes  
- Enables ADAS-level safety awareness through audio-visual alerts  

# Table of Contents
- Demo Photos
- Libraries
- Block Diagram
- Code Base
- Technologies Used
- Results
- Conclusion
- Future Scope


## Demo Photos
| YOLOv11 Detection | TSNet Classification |
|-------------------|----------------------|
| ![Detection](<img width="1072" height="667" alt="Screenshot 2025-09-07 215205" src="https://github.com/user-attachments/assets/5f53b092-10ff-4fff-b655-bc6ba0bc70ac" />) | ![Classification](<img width="1919" height="1006" alt="Screenshot 2025-08-10 161416" src="https://github.com/user-attachments/assets/11168168-427a-48a5-875b-872bd056488b" />)|

(Sample outputs demonstrating real-time detection and lassification.) <img width="1080" height="638" alt="Screenshot 2025-09-07 164738" src="https://github.com/user-attachments/assets/cf61ea49-feb2-4d36-910c-9b7e39c7cf38" />

## Libraries

| Library | Description |
|----------|--------------|
| **YOLOv11s-seg** | Detects traffic sign regions from real-time video input |
| **TSNet (Custom CNN)** | Lightweight CNN for classifying detected traffic signs |
| **OpenCV** | Handles video capture, preprocessing, and visualization |
| **TensorFlow / Keras** | Framework for model design and training |
| **NumPy / Pandas** | Data processing and numerical operations |
| **cvzone** | Simplifies computer vision overlays and display elements |
| **scikit-learn** | Used for preprocessing, evaluation, and model metrics |
| **pyttsx3** | Provides text-to-speech driver alerts |
| **Matplotlib** | Visualizes accuracy, loss curves, and confusion matrices |

# Dataset Details
- **Source:** GTSRB (German Traffic Sign Recognition Benchmark) + Indian Traffic Signs Dataset  
- **Total Images:** 42,235  
- **Number of Classes:** 47  
- **Data Split:** 80% Training / 20% Validation  

# Block Diagram
<img width="940" height="1067" alt="image" src="https://github.com/user-attachments/assets/559f8148-565a-45a1-a9e6-53a3c542228c" />


# Code Base
# 1. Model Architecture
- **YOLOv11s-seg:** Trained for region detection of traffic signs.
- **TSNet:** Two-block CNN with Conv2D → MaxPooling → Dense(Softmax) for 47-class classification.

### 2. Real-Time Implementation
- Captures live video using **OpenCV**.
- Applies YOLO detection and TSNet classification per frame.
- Displays labels, bounding boxes, and confidence scores.
- Provides **voice alerts** using pyttsx3.

#  Technologies Used
| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow, Keras |
| Object Detection | YOLOv11s-seg |
| Image Processing | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, cvzone |
| Voice Alert | pyttsx3 |
| IDE | PyCharm |

# Results
| Metric | Detection (YOLOv11s-seg) | Classification (TSNet) |
|---------|--------------------------|--------------------------|
| Precision | 0.91 | 0.98 |
| Recall | 0.90 | 0.97 |
| Accuracy | 92% | **98.1%** |

# Highlights:
- Accurate detection of multiple signs in complex traffic scenes.  
- Robust to varied lighting, occlusions, and angles.  
- Real-time processing at ~30 FPS.  
- Seamless integration of **visual + voice alerts**.

# Conclusion
**Project RTS** successfully demonstrates an efficient, real-time traffic sign detection system using **Deep Learning** and **Computer Vision**.  
The system’s dual-stage pipeline (YOLOv11 + TSNet) enhances accuracy and reliability while maintaining low latency, enabling real-time operation suitable for ADAS and smart city applications.  

**Key Takeaways:**
- Combines detection and classification for optimal accuracy.  
- Achieves 98.1% recognition on real-world data.  
- Lightweight, scalable, and deployable on embedded systems.  


##  Future Scope
- Integration with **Lane Detection** and **Driver Drowsiness Modules**.  
- Deployment on **Edge Devices** (Jetson Nano, Raspberry Pi).  
- Inclusion of **region-specific datasets** for global adaptability.  
- Implementation of **voice-assisted driver alerts** and **HUD displays**.  
- Exploration of **transformer-based architectures (e.g., DETR)** for next-gen models.

“Driving Towards Safer, Smarter Roads – Project RTS”
