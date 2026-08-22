from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass


from config import DEVICE


@dataclass
class SegmentationResult:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    class_id: int
    class_name: str
    centroid: tuple[int, int]
    area: float


class InstanceSegmentor:
    def __init__(self, model_path: str, conf: float = 0.4, device: str = DEVICE):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def segment(self, frame: np.ndarray, detections: list) -> list[SegmentationResult]:
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        seg_results = []

        for r in results:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes
            if boxes is None:
                continue

            for i, mask_raw in enumerate(masks):
                cls_id = int(boxes.cls[i])
                h, w = frame.shape[:2]
                mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                largest = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest)
                area = cv2.contourArea(largest)
                cx = x + bw // 2
                cy = y + bh // 2

                seg_results.append(SegmentationResult(
                    mask=mask_binary,
                    bbox=(x, y, x + bw, y + bh),
                    class_id=cls_id,
                    class_name=r.names[cls_id],
                    centroid=(cx, cy),
                    area=area,
                ))

        return seg_results

    def overlay(self, frame: np.ndarray, results: list[SegmentationResult], alpha: float = 0.4) -> np.ndarray:
        overlay = frame.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for i, res in enumerate(results):
            color = colors[i % len(colors)]
            mask_3ch = np.stack([res.mask] * 3, axis=-1)
            color_layer = np.full_like(frame, color, dtype=np.uint8)
            blended = cv2.addWeighted(frame, 1 - alpha, color_layer, alpha, 0)
            overlay = np.where(mask_3ch, blended, overlay)

        return overlay
