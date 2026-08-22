from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

from config import VEHICLE_CLASSES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, LaneConfig, DEVICE


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    centroid: tuple[int, int]


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
            self.model.to(self.device)
            try:
                self.model.model.half()
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
            boxes = r.boxes
            if boxes is None:
                continue
            names = getattr(r, "names", getattr(self.model, "names", VEHICLE_CLASSES))
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, VEHICLE_CLASSES.get(cls_id, f"cls_{cls_id}"))
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not self._is_inside_regions((x1, y1, x2, y2), (cx, cy)):
                    continue

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                    centroid=(cx, cy),
                ))
        return detections
