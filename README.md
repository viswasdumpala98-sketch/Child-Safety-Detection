# 🧒🔍 Child Safety Detection System

This project is a **real-time harmful object detection system** designed to monitor environments around children. It detects potentially dangerous objects like **batteries** and **candles** using computer vision and deep learning.

Built with **YOLOv8**, **Gradio**, and **OpenCV**, it supports:

- 🖼️ Image detection
- 📹 Video detection
- 🎥 Live webcam detection

---

## 🚀 Features

- ✅ Real-time object detection
- ✅ Trained on custom dataset with 6 hazardous/non-hazardous classes
- ✅ Web interface using Gradio
- ✅ Status alerts: **DANGER** or **Safe**
- ✅ Extracts and saves cropped dangerous objects from live webcam feed
- ✅ Exported YOLOv8 model (`best.pt`) included

---

## 📂 Project Structure

```
children_safety/
├── app.py # Gradio-based interface
├── predict.py # Image/video prediction logic
├── realtime_webcam.py # Live webcam detection
├── training.py # YOLOv8 training script
├── split_dataset.py # Data split helper
├── yolo_convert.py # YOLO format converter (if used)
├── data.yaml # YOLO dataset config
├── .gitignore
├── requirements.txt
└── runs/...
```

---

## 🔍 Supported Object Classes

1. battery ⚠️
2. candle ⚠️
3. toycar
4. dice ⚠️
5. spoon
6. highlighter

> ⚠️ `battery`, `candle` and `dice` are flagged as **DANGER** items.

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/Nihitha-S/child-safety-detection.git
cd child-safety-detection
```

Create and activate a virtual environment (optional but recommended):
```
python3.10 -m venv venv310
venv310\Scripts\activate  # on Windows
```

## Install dependencies and run:
```
pip install -r requirements.txt
```
and run
```
python app.py
```

## How to Use
- 📷 Image Detection
  - Navigate to "Upload Image" tab

  - Upload image → Click "Detect Objects"

- 🎥 Video Detection
  - Go to "Upload Video" tab

  - Upload .mp4 file → Click "Detect Objects"

- 🔴 Live Webcam Detection
  - Go to "Live Webcam" tab

  - Click "Launch Live Webcam Detection"

Press 'q' to stop the web cam

- 📸 Sample Output
  - Detected image with danger warning overlay

## Model Training
  
  To retrain or fine-tune the model:
```
python training.py
```
   - Uses data.yaml to point to the dataset

   - Default: yolov8n.pt trained for 20 epochs

- Outputs saved in runs/

### Trained Model

The model file best.pt is stored at:

`runs/detect/child_safety_v1/weights/best.pt`

Used for all inference modes in this project

## Author
- SreeHarshini
- GitHub: [@Harshini V](https://github.com/harshin-i)

## 📄 License
This project is licensed under the MIT License.
You’re free to modify, distribute, or use it in any child-safety or academic context.
