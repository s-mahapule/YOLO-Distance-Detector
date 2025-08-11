
# StreetVision – Real-Time Object Detection with Distance & Voice Alerts

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv4-Tiny](https://img.shields.io/badge/YOLOv4--Tiny-Enabled-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

Real-time YOLOv4-Tiny object detection with distance estimation, voice alerts, and a Tkinter control panel.

**StreetVision** is a Python-based computer vision tool that uses **YOLOv4-Tiny** to detect objects in real-time from your camera, estimate their distance, and announce them with voice alerts.  
It features a simple **Tkinter GUI** to switch between detection modes, adjust confidence, and toggle voice output.

---

## ✨ Features
- **Real-time YOLOv4-Tiny object detection** (CPU-based, no GPU required)
- **Distance estimation** for each detected object
- **Voice alerts** that speak object name, position, and distance
- **GUI controls** to:
  - Switch between **Street**, **Animals**, or **All** detection modes
  - Adjust detection confidence
  - Change resolution
  - Enable/disable voice output
    
## 📥 Setup

Download YOLOv4-Tiny weights
The YOLOv4-Tiny config file (yolov4-tiny.cfg) is already included in this repository.
However, the weights file is not included because it is large.

Official download link:
📥 [Download yolov4-tiny.weights (from AlexeyAB/darknet)](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

After downloading:
Place the file yolov4-tiny.weights in the same folder as StreetVision.py.

Run the application:
python detector_gui.py

---

## 📦 Requirements
Make sure you have **Python 3.8+** installed.  
Install dependencies with:
```bash
pip install -r requirements.txt
