from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field

from src.tracking.multi_object_tracker import Track


@dataclass
class ClassifiedVehicle:
    track_id: int
    vehicle_type: str
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    confidence: float
    direction: str = ""
    lane: str = ""
    speed: float = 0.0
    history: list = field(default_factory=list)

    @property
    def class_name(self) -> str:
        return self.vehicle_type


VEHICLE_SIZE_RATIOS = {
    "car": (0.8, 3.0),
    "motorcycle": (0.3, 1.5),
    "bus": (1.5, 5.0),
    "truck": (1.2, 5.0),
    "auto_rickshaw": (0.4, 2.0),
    "bicycle": (0.2, 1.2),
    "van": (1.0, 4.0),
    "minivan": (0.9, 3.5),
    "suv": (0.9, 3.5),
}


class VehicleClassifier:
    def __init__(self):
        self.class_names = {
            0: "car", 1: "motorcycle", 2: "bus", 3: "truck",
            4: "auto_rickshaw", 5: "bicycle", 7: "van", 8: "minivan", 9: "suv",
        }

    def classify(self, track: Track, fps: float = 25.0) -> ClassifiedVehicle:
        speed = self.estimate_speed(track, fps)
        return ClassifiedVehicle(
            track_id=track.track_id,
            vehicle_type=track.class_name,
            bbox=track.bbox,
            centroid=track.centroid,
            confidence=track.confidence,
            speed=speed,
            history=list(track.history),
        )

    def classify_batch(self, tracks: list[Track], fps: float = 25.0) -> list[ClassifiedVehicle]:
        return [self.classify(t, fps) for t in tracks]

    def estimate_speed(self, track: Track, fps: float, pixels_per_meter: float = 10.0) -> float:
        if len(track.history) < 2 or fps <= 0:
            return 0.0
        dx = track.history[-1][0] - track.history[-3][0] if len(track.history) >= 3 else track.history[-1][0] - track.history[0][0]
        dy = track.history[-1][1] - track.history[-3][1] if len(track.history) >= 3 else track.history[-1][1] - track.history[0][1]
        dist_px = (dx**2 + dy**2) ** 0.5
        frames = 3 if len(track.history) >= 3 else 2
        dist_m = dist_px / pixels_per_meter
        speed = dist_m * fps / frames
        return round(speed * 3.6, 1)  # m/s to km/h
