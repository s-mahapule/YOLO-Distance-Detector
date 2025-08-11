
# StreetVision – Real-Time Object Detection with Distance & Voice Alerts

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

---

## 📦 Requirements
Make sure you have **Python 3.8+** installed.  
Install dependencies with:
```bash
pip install -r requirements.txt
