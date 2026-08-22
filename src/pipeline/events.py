from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = None
    crop: np.ndarray | None = None

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def bottom_center(self) -> tuple[int, int]:
        """Tire/ground contact point on the road plane."""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int(y2))


@dataclass
class TrafficViolation:
    track_id: int
    violation_type: str  # "WRONG_WAY" or "CYCLE_ZONE_INTRUSION"
    description: str
    vehicle_type: str
    lane: str
    timestamp: float
    frame_idx: int


@dataclass
class TrafficEvent:
    timestamp: float
    frame_idx: int
    event_type: str  # "VEHICLE_TRACKED", "WRONG_WAY", "CYCLE_ZONE_INTRUSION"
    track_id: int
    vehicle_type: str
    lane: str
    direction: str
    plate_text: str = ""
    speed_kmh: float = 0.0
    confidence: float = 1.0
    bbox: str = ""
    crop_path: str = ""
