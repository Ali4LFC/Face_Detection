"""
app/config.py — Central path & detection configuration.

All other modules import paths from here so that relocating
the project only requires changing ROOT_DIR.
"""

from pathlib import Path

# ── Root directories ──────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent   # .../проект/
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR   = ROOT_DIR / "data"

# ── Model files ───────────────────────────────────────────────
YOLO_MODEL       = MODELS_DIR / "yolov8s.pt"
YOLO_MODEL_NANO  = MODELS_DIR / "yolov8n.pt"
YOLO_POSE_MODEL  = MODELS_DIR / "yolov8n-pose.pt" # Lightweight pose model
COCO_NAMES       = MODELS_DIR / "coco.names"

# ── Detection thresholds (defaults) ───────────────────────────
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD        = 0.40
INPUT_SIZE           = 640           # Optimized for speed

# ── Action Detection thresholds ───────────────────────────────
ACTION_HANDS_UP_THRESH = 0.1         # Height difference for hands up
ACTION_FALLING_RATIO   = 0.6         # Width/Height ratio to detect falling

# ── COCO class IDs ────────────────────────────────────────────
PERSON_CLASS_ID = 0
DANGEROUS_CLASS_IDS: dict[int, str] = {
    43: "knife",
    76: "scissors",
    34: "baseball bat",
}

# ── Detection mode ────────────────────────────────────────────
# If True: show ALL objects from COCO (80 classes). If False: only dangerous objects.
DETECT_ALL_COCO_OBJECTS = True

# ── Camera settings ──────────────────────────────────────────
# CAMERA_INDEX  — single camera index used by the main app:
#   0 = built-in laptop camera
#   1 = USB camera  ← currently selected
#  -1 = auto-detect (scans indices 1-4, then 0)
CAMERA_INDEX        = -1    # -1 = auto-detect all cameras
CAMERA_PREFER_USB   = True   # used only when CAMERA_INDEX = -1
CAMERA_INDICES      = [1]    # for any multi-camera features
MAX_CAMERAS         = 4
