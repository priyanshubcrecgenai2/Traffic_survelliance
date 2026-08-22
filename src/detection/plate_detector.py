from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

from config import MODELS_DIR, CONFIDENCE_THRESHOLD, DEVICE


@dataclass
class PlateDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    text: str = ""
    crop: np.ndarray | None = None


class PlateDetector:
    def __init__(self, model_path: str = str(MODELS_DIR / "license-plate-finetune-v1l.pt"), conf: float = CONFIDENCE_THRESHOLD, device: str = DEVICE):
        self.conf = conf
        self.device = device
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, vehicle_bboxes: list[tuple[int, int, int, int]] | None = None) -> list[PlateDetection]:
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
                plates.append(PlateDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                ))

        if vehicle_bboxes:
            filtered = []
            for plate in plates:
                px1, py1, px2, py2 = plate.bbox
                pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                for vx1, vy1, vx2, vy2 in vehicle_bboxes:
                    if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                        plate.crop = frame[py1:py2, px1:px2].copy()
                        filtered.append(plate)
                        break
            plates = filtered

        for plate in plates:
            if plate.crop is None:
                x1, y1, x2, y2 = plate.bbox
                plate.crop = frame[y1:y2, x1:x2].copy()

        return plates
