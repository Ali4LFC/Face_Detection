# AI Security Monitoring System (MVP) 🚀

[Русская версия (Russian Version)](./README_RU.md)

An intelligent video surveillance system powered by **YOLOv8**, designed for automatic person detection, skeletal analysis (pose estimation), and multi-camera monitoring.

---

## Key Features 🛠

- **Skeleton Detection (Pose Estimation)**: Uses `YOLOv8-pose` model to identify 17 body keypoints in real-time.
- **Multi-Camera Mode**: Supports simultaneous display of multiple video streams in a grid (Grid View).
- **Auto-Camera Discovery**: Automatically detects connected USB cameras, prioritizing external devices over built-in ones.
- **Performance Optimization**:
    - Uses lightweight `nano` models for high FPS.
    - Optimized processing resolution (640px).
    - Multi-threaded video processing for a smooth UI.
- **Modern GUI**: Dark theme built with `CustomTkinter`.

---

## Project Structure 📂

```text
project/
├── app/
│   ├── core/           # Detection logic (YOLO, OpenCV)
│   ├── ui/             # Graphical user interface (CustomTkinter)
│   └── config.py       # Centralized configuration
├── models/             # Directory for neural network weights (*.pt)
├── assets/             # Test videos and images
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── .gitignore          # Git exclusion rules
```

---

## Installation & Setup 🚀

### 1. Clone the repository
```bash
git clone https://github.com/vash-username/security-intelligence-system.git
cd security-intelligence-system
```

### 2. Install dependencies
Virtual environment is recommended.
```bash
pip install -r requirements.txt
```

### 3. Prepare models
On first run, the app will automatically download the necessary `yolov8n-pose.pt` weights to the `models/` folder. You can also download them manually from the [Ultralytics repository](https://github.com/ultralytics/ultralytics).

### 4. Run
```bash
python main.py
```

---

## Multi-Camera Configuration 📹

To change the list of cameras, open `app/config.py` and edit the following parameter:
```python
CAMERA_INDICES = [0, 1]  # Your camera indices
```
The system will automatically adjust the grid layout based on the number of cameras.

---

## Tech Stack 💻

- **Python 3.10+**
- **Ultralytics YOLOv8** (Object Detection & Pose)
- **OpenCV** (Video Processing)
- **CustomTkinter** (Modern UI)
- **Pillow** (Image Handling)

---

## Author 👤

**Ali** - *Development and Integration*
**Ayaulym&** - *Design and Development*

---
*Project developed as part of the "Computer Information Processing" course (Zhukabayeva T. K.)*
