Alright 🚀
Here’s your **fully upgraded GitHub-ready README** — with badges, emojis, screenshot section, and a clean, modern structure.
Just replace your current `README.md` content with this:

---


# 🚦 StreetVision – Real-Time Object Detection with Distance & Voice Alerts  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv4-Tiny](https://img.shields.io/badge/YOLOv4--Tiny-Enabled-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**StreetVision** is a Python-based computer vision tool that uses **YOLOv4-Tiny** to detect objects in real-time from your camera, estimate their distance, and announce them with voice alerts.  
It features a **smooth Tkinter GUI** to switch between detection modes, adjust confidence, and toggle voice output.

---

## ✨ Features
✅ Real-time **YOLOv4-Tiny** object detection (CPU-friendly)  
✅ **Distance estimation** for each detected object  
✅ **Voice alerts** for object name, position & distance  
✅ **GUI controls** for mode, confidence, resolution, and voice  
✅ Detects **street objects**, **animals**, or **all objects**  

---

## 📦 Requirements
- Python **3.8+**
- Webcam or USB camera

**Install dependencies:**
```bash
pip install -r requirements.txt
````
---

## 📥 Setup

1️⃣ **Clone the repository:**

```bash
git clone https://github.com/s-mahapule/YOLO-Distance-Detector.git
cd YOLO-Distance-Detector
```

2️⃣ **Download YOLOv4-Tiny weights**

* Config file (`yolov4-tiny.cfg`) is already included.
* Download the official weights here:
  [📥 Download yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
* Place `yolov4-tiny.weights` in the **same folder** as your Python file.

3️⃣ **Run the app:**

```bash
python StreetVision.py
```

---

## 🎮 GUI Controls

| Button / Option | Description                                    |
| --------------- | ---------------------------------------------- |
| **Mode**        | Choose `Street`, `Animals`, or `All` detection |
| **Conf**        | Adjust detection confidence threshold          |
| **Res**         | Set YOLO input resolution (416 or 608)         |
| **Voice**       | Enable/disable voice alerts                    |
| **Start**       | Start detection                                |
| **Stop**        | Stop detection                                 |

Press **`q`** in the detection window to stop the camera.

---

## ⚠️ Notes

* Detection accuracy depends on lighting, camera quality, and object visibility
* Distance estimation is **approximate**
* CPU mode is used by default; you can enable GPU in the code if supported

---

## 📄 License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute.

---

## 👩‍💻 Author

Developed by **[Shafia Mahapule](https://github.com/s-mahapule)**
Inspired by YOLOv4-Tiny and OpenCV distance estimation techniques.

