"""
Modular License Plate Detection & OCR Subsystem
Note: Modular and optional. Disabled by default on wide-angle overview cameras.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO

from config import MODELS_DIR, CONFIDENCE_THRESHOLD, DEVICE


@dataclass
class PlateResult:
    bbox: tuple[int, int, int, int]
    confidence: float
    text: str = ""
    crop: np.ndarray | None = None


class PlateDetector:
    def __init__(
        self,
        model_path: str = str(MODELS_DIR / "license-plate-finetune-v1l.pt"),
        conf: float = CONFIDENCE_THRESHOLD,
        device: str = DEVICE,
    ):
        self.conf = conf
        self.device = device
        self.model = YOLO(model_path) if Path(model_path).is_file() else None

    def detect(self, frame: np.ndarray) -> list[PlateResult]:
        if self.model is None:
            return []

        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        plates = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2].copy()
                plates.append(PlateResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    crop=crop,
                ))
        return plates
