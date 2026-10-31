"""YOLO wrapper for detecting 5 stationery classes.

Loads custom-trained weights (models/best.pt) when available. If they are
missing, falls back to the stock COCO model, which can only detect scissors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# The 5 target classes, in the same order as data.yaml.
CLASSES = ["Pen", "Pencil", "Eraser", "Ruler", "Scissors"]

CLASS_NAMES_TH = {
    "Pen": "ปากกา",
    "Pencil": "ดินสอ",
    "Eraser": "ยางลบ",
    "Ruler": "ไม้บรรทัด",
    "Scissors": "กรรไกร",
}

# BGR colors for drawing boxes.
CLASS_COLORS = {
    "Pen": (255, 128, 0),
    "Pencil": (0, 200, 255),
    "Eraser": (203, 102, 255),
    "Ruler": (0, 220, 0),
    "Scissors": (60, 60, 255),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "models", "best.pt")
FALLBACK_WEIGHTS = "yolov8n.pt"  # COCO; downloaded automatically by ultralytics


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


def _canonical(name: str) -> str | None:
    """Map a model's class name (any case) to one of CLASSES, or None."""
    lookup = {c.lower(): c for c in CLASSES}
    return lookup.get(str(name).strip().lower())


class StationeryDetector:
    def __init__(self, weights: str | None = None, device: str | None = None):
        from ultralytics import YOLO  # imported lazily so the GUI opens fast

        weights = weights or DEFAULT_WEIGHTS
        self.using_fallback = False
        if not os.path.exists(weights):
            weights = FALLBACK_WEIGHTS
            self.using_fallback = True

        self.weights = weights
        self.model = YOLO(weights)
        self.device = device

        # model class id -> our canonical class name (others are ignored)
        self.class_map: dict[int, str] = {}
        for idx, name in self.model.names.items():
            canon = _canonical(name)
            if canon:
                self.class_map[int(idx)] = canon

    @property
    def supported_classes(self) -> list[str]:
        return [c for c in CLASSES if c in self.class_map.values()]

    def detect(self, frame: np.ndarray, conf: float = 0.5, iou: float = 0.45) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=conf,
            iou=iou,
            classes=list(self.class_map.keys()) or None,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for xyxy, c, k in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            name = self.class_map.get(int(k))
            if name is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            detections.append(Detection(name, float(c), (x1, y1, x2, y2)))
        return detections


def count_by_class(detections: list[Detection]) -> dict[str, int]:
    counts = {c: 0 for c in CLASSES}
    for d in detections:
        counts[d.cls] += 1
    return counts


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and 'Class 0.87' labels on a copy of the frame."""
    out = frame.copy()
    scale = max(0.6, out.shape[1] / 1280 * 0.9)  # keep labels readable on large frames
    thick = max(2, round(scale * 2.5))
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = CLASS_COLORS.get(d.cls, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        label = f"{d.cls} {d.conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ty = max(y1, th + base + 4)
        cv2.rectangle(out, (x1, ty - th - base - 6), (x1 + tw + 10, ty), color, -1)
        cv2.putText(out, label, (x1 + 5, ty - base - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return out
