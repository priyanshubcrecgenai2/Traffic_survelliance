from __future__ import annotations

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from config import LaneConfig, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, DEVICE
from src.pipeline.events import Detection


DEFAULT_CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "auto_rickshaw",
    5: "bicycle",
    6: "pedestrian",
    7: "van",
    8: "minivan",
    9: "suv",
}


class VehicleDetector:
    def __init__(
        self,
        model_path: str,
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        device: str = DEVICE,
        regions: list[LaneConfig] | None = None,
        zones: dict | None = None,
    ):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.regions = regions or []
        self.zones = zones or {}
        
        self.model = YOLO(model_path)
        if "cuda" in str(self.device):
            try:
                self.model.to(self.device)
                self.model.model.half()  # Enable Tensor Core FP16
            except Exception:
                pass

    def set_regions(self, regions: list[LaneConfig], zones: dict | None = None):
        self.regions = regions
        if zones is not None:
            self.zones = zones

    def _is_inside_regions(self, bbox: tuple[int, int, int, int], centroid: tuple[int, int]) -> bool:
        if not self.regions and not self.zones:
            return True

        cx, cy = centroid
        x1, y1, x2, y2 = bbox
        bottom_mid = (cx, y2)
        pts_to_check = [centroid, bottom_mid, (cx, (cy + y2) // 2)]

        # Check lane polygons
        for lane in self.regions:
            if lane.polygon and len(lane.polygon) >= 3:
                poly_np = np.array(lane.polygon, dtype=np.int32)
                for pt in pts_to_check:
                    if cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0:
                        return True
            elif lane.x_start < lane.x_end:
                if lane.x_start <= cx <= lane.x_end:
                    return True

        # Check monitoring zones
        for z_val in self.zones.values():
            if isinstance(z_val, (list, tuple)):
                if len(z_val) == 4 and all(isinstance(v, (int, float)) for v in z_val):
                    zx1, zy1, zx2, zy2 = map(int, z_val)
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        return True
                elif len(z_val) >= 3:
                    poly_np = np.array(z_val, dtype=np.int32)
                    for pt in pts_to_check:
                        if cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False) >= 0:
                            return True
        return False

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=640,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cls_name = r.names.get(cls_id, DEFAULT_CLASS_NAMES.get(cls_id, f"cls_{cls_id}"))
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if self._is_inside_regions((x1, y1, x2, y2), centroid):
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    ))

        return detections
