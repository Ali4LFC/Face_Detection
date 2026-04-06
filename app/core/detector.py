"""
app/core/detector.py — Advanced Detection & Action Analysis.

Features:
  - Object Detection (YOLOv8s) for weapons.
  - Pose Estimation (YOLOv8-pose) for skeletal analysis.
  - Action Detection: Falling, Hands Up.
"""

import cv2
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from ultralytics import YOLO

from app.config import (
    CONFIDENCE_THRESHOLD,
    PERSON_CLASS_ID,
    DANGEROUS_CLASS_IDS,
    YOLO_MODEL,
    YOLO_POSE_MODEL,
    COCO_NAMES,
    INPUT_SIZE,
    ACTION_HANDS_UP_THRESH,
    ACTION_FALLING_RATIO,
    DETECT_ALL_COCO_OBJECTS,
)

# ──────────────────────────────────────────────────────────────
# Visualization Settings
# ──────────────────────────────────────────────────────────────

COLOR_PERSON    = (0, 200, 100)
COLOR_SKELETON  = (255, 255, 0)
COLOR_KEYPOINT  = (0, 0, 255)
COLOR_DANGER    = (0, 0, 255)
COLOR_ACTION    = (255, 100, 0) # Orange for actions
COLOR_ALERT_BG  = (0, 0, 180)
COLOR_SAFE_BG   = (0, 140, 0)
COLOR_TEXT      = (255, 255, 255)

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single detected region."""
    class_id:     int
    class_name:   str
    confidence:   float
    x: int; y: int; w: int; h: int
    is_dangerous: bool = False
    keypoints:    np.ndarray = None
    actions:      List[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    """Analysis results for a single frame."""
    detections:         List[Detection] = field(default_factory=list)
    person_count:       int = 0
    danger_detected:    bool = False
    dangerous_actions:  List[str] = field(default_factory=list)
    dangerous_objects:  List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ──────────────────────────────────────────────────────────────
# Object Detector Class
# ──────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Combined Detector for Objects, Poses, and Actions.
    """

    def __init__(self):
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.privacy_mode = False
        
        # Using Nano models for speed
        from app.config import YOLO_MODEL_NANO, YOLO_POSE_MODEL
        
        print("[AI] Loading Object Model (Nano)...")
        self.model_obj = YOLO(str(YOLO_MODEL_NANO))
        
        print("[AI] Loading Pose Model...")
        self.model_pose = YOLO(str(YOLO_POSE_MODEL))

        # COCO class names (for readable labels)
        self.coco_names: list[str] = []
        try:
            self.coco_names = [ln.strip() for ln in COCO_NAMES.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            # If file is missing/encoding issues, fall back to Ultralytics internal names
            self.coco_names = []

    def detect(self, image: np.ndarray) -> DetectionResult:
        tick = cv2.getTickCount()

        # 1. Detect Objects (COCO)
        # For MVP, allow full COCO detection; optionally can be restricted to dangerous classes.
        obj_kwargs = dict(imgsz=INPUT_SIZE, conf=self.confidence_threshold, verbose=False)
        if not DETECT_ALL_COCO_OBJECTS:
            obj_kwargs["classes"] = list(DANGEROUS_CLASS_IDS.keys())
        results_obj = self.model_obj(image, **obj_kwargs)

        # 2. Detect Poses (People skeletons)
        results_pose = self.model_pose(
            image, imgsz=INPUT_SIZE, conf=self.confidence_threshold, verbose=False
        )

        elapsed = (cv2.getTickCount() - tick) / cv2.getTickFrequency() * 1000
        result = DetectionResult(processing_time_ms=elapsed)

        # Process Objects
        if results_obj:
            for box in results_obj[0].boxes:
                cid = int(box.cls[0])
                # People are handled by pose model (skeleton + actions) to avoid duplicates.
                if cid == PERSON_CLASS_ID:
                    continue
                if self.coco_names and 0 <= cid < len(self.coco_names):
                    cname = self.coco_names[cid]
                else:
                    # Ultralytics model has its own names mapping
                    cname = getattr(self.model_obj, "names", {}).get(cid, str(cid))
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                det = Detection(
                    class_id=cid, class_name=cname, confidence=float(box.conf[0]),
                    x=x1, y=y1, w=x2-x1, h=y2-y1, is_dangerous=(cid in DANGEROUS_CLASS_IDS)
                )
                result.detections.append(det)
                if det.is_dangerous:
                    result.danger_detected = True
                    result.dangerous_objects.append(DANGEROUS_CLASS_IDS.get(cid, cname))

        # Process Poses & Actions
        if results_pose:
            kpts = results_pose[0].keypoints
            boxes = results_pose[0].boxes
            
            if kpts is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    kp = kpts.xy[i].cpu().numpy()
                    
                    det = Detection(
                        class_id=PERSON_CLASS_ID, class_name="person", 
                        confidence=float(box.conf[0]),
                        x=x1, y=y1, w=x2-x1, h=y2-y1, keypoints=kp
                    )
                    
                    # Analyze Actions
                    actions = self._analyze_pose_actions(det)
                    det.actions = actions
                    if actions:
                        result.danger_detected = True
                        result.dangerous_actions.extend(actions)
                    
                    result.detections.append(det)
                    result.person_count += 1

        return result

    def _analyze_pose_actions(self, det: Detection) -> List[str]:
        """Heuristic analysis of pose keypoints to detect actions."""
        actions = []
        kp = det.keypoints
        if kp is None or len(kp) < 17: return actions

        # Hands Up (wrists above nose by a relative threshold)
        # Keypoints: 0=nose, 9=left_wrist, 10=right_wrist
        nose_y = kp[0][1]
        l_wrist_y = kp[9][1]
        r_wrist_y = kp[10][1]
        if nose_y > 0:
            margin = max(20.0, det.h * (0.15 + ACTION_HANDS_UP_THRESH))
            if (l_wrist_y > 0 and l_wrist_y < nose_y - margin) or (r_wrist_y > 0 and r_wrist_y < nose_y - margin):
                actions.append("HANDS UP")

        # Falling Detection (Heuristic: Width > Height * 1.5 or horizontal alignment)
        # We check aspect ratio and if the head is low relative to the body
        aspect_ratio = det.w / max(1.0, det.h)
        if aspect_ratio > 1.4:
            # Check if person is on the ground
            # Using hips (11, 12) vs nose (0) or ankles (15, 16)
            l_hip_y, r_hip_y = kp[11][1], kp[12][1]
            if (l_hip_y > 0 and abs(nose_y - l_hip_y) < det.h * 0.4) or (r_hip_y > 0 and abs(nose_y - r_hip_y) < det.h * 0.4):
                actions.append("FALLING")

        return actions

    def draw_detections(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        canvas = image.copy()
        h, w = canvas.shape[:2]

        # ── Extract weapon centres for proximity check ─────────
        weapon_centres = []
        person_boxes   = []
        for det in result.detections:
            cx = det.x + det.w // 2
            cy = det.y + det.h // 2
            if det.is_dangerous:
                weapon_centres.append((cx, cy, max(det.w, det.h)))
            else:
                person_boxes.append((det.x, det.y, det.x + det.w, det.y + det.h))

        # ── Improvement 2: Threat Zone circles around weapons ──
        for (wcx, wcy, wsize) in weapon_centres:
            radius = int(wsize * 1.8)  # danger radius = 1.8× weapon size

            # Check if any person is inside the circle
            person_inside = False
            for (px1, py1, px2, py2) in person_boxes:
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2
                dist = ((pcx - wcx)**2 + (pcy - wcy)**2) ** 0.5
                if dist < radius:
                    person_inside = True
                    break

            zone_color = (0, 0, 255) if person_inside else (0, 80, 200)

            # Semi-transparent filled circle (OpenCV addWeighted)
            overlay = canvas.copy()
            cv2.circle(overlay, (wcx, wcy), radius, zone_color, -1)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            # Dashed border circle
            cv2.circle(canvas, (wcx, wcy), radius, zone_color, 2)

            if person_inside:
                warn_label = "! PROXIMITY ALERT"
                cv2.putText(canvas, warn_label, (wcx - 70, wcy - radius - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # ── Draw each detection ────────────────────────────────
        for det in result.detections:
            color = COLOR_DANGER if det.is_dangerous else COLOR_PERSON
            if det.actions: color = COLOR_ACTION

            x1, y1 = max(0, det.x), max(0, det.y)
            x2, y2 = min(w, det.x + det.w), min(h, det.y + det.h)

            if det.keypoints is not None:
                self._draw_skeleton(canvas, det.keypoints)
                if self.privacy_mode:
                    self._blur_face(canvas, det.keypoints)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name.upper()}"
            if det.actions:
                label += f" | {' & '.join(det.actions)}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        # ── Improvement 1: Pulsing red border when danger ─────
        if result.danger_detected:
            self._draw_pulse_border(canvas)

        return self._draw_info_panel(canvas, result)

    @staticmethod
    def _draw_pulse_border(image: np.ndarray):
        """
        Improvement 1 — Pulsing red border alarm.
        Uses time.time() + sin() to vary opacity, creating a heartbeat effect.
        """
        h, w = image.shape[:2]
        # Pulse: opacity oscillates between 0.15 and 0.65 at ~2 Hz
        pulse = 0.4 + 0.25 * np.sin(time.time() * 6.0)
        thickness = max(6, int(w * 0.012))  # scale thickness to image size

        overlay = image.copy()
        # Draw 3 nested rectangles for a glow effect
        for shrink in range(3):
            s = shrink * (thickness // 3)
            cv2.rectangle(overlay, (s, s), (w - s, h - s), (0, 0, 255), thickness - shrink * 2)
        cv2.addWeighted(overlay, pulse, image, 1.0 - pulse, 0, image)

    def _draw_skeleton(self, image: np.ndarray, kp: np.ndarray):
        for start, end in SKELETON_CONNECTIONS:
            pt1 = tuple(kp[start].astype(int))
            pt2 = tuple(kp[end].astype(int))
            if pt1[0] > 0 and pt2[0] > 0:
                cv2.line(image, pt1, pt2, COLOR_SKELETON, 2)
        for x, y in kp:
            if x > 0: cv2.circle(image, (int(x), int(y)), 3, COLOR_KEYPOINT, -1)

    def _blur_face(self, image: np.ndarray, kp: np.ndarray):
        """Blur face area using pose keypoints (0-4: nose, eyes, ears)."""
        face_kpts = kp[:5]
        valid_face = face_kpts[face_kpts[:, 0] > 0]
        if len(valid_face) < 2: return

        # Calculate bounding box for the face area
        fx1, fy1 = np.min(valid_face, axis=0).astype(int)
        fx2, fy2 = np.max(valid_face, axis=0).astype(int)
        
        # Add padding
        pad = int(max(fx2 - fx1, fy2 - fy1) * 0.5)
        fx1, fy1 = max(0, fx1 - pad), max(0, fy1 - pad)
        fx2, fy2 = min(image.shape[1], fx2 + pad), min(image.shape[0], fy2 + pad)

        if fx2 > fx1 and fy2 > fy1:
            face_roi = image[fy1:fy2, fx1:fx2]
            # Strong blur for privacy
            blurred = cv2.GaussianBlur(face_roi, (51, 51), 0)
            image[fy1:fy2, fx1:fx2] = blurred

    def _draw_info_panel(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        panel_h = 80
        overlay = image.copy()
        bg = COLOR_ALERT_BG if result.danger_detected else COLOR_SAFE_BG
        cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_h), bg, -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        
        status = "SAFE"
        if result.dangerous_objects or result.dangerous_actions:
            status = "WARNING: " + ", ".join(result.dangerous_objects + result.dangerous_actions)
        
        cv2.putText(image, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        stats = f"People: {result.person_count} | Latency: {result.processing_time_ms:.0f}ms"
        cv2.putText(image, stats, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return image
