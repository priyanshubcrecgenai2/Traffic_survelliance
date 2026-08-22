from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from config import LaneConfig


import cv2
import numpy as np


@dataclass
class LaneCount:
    lane_name: str
    direction: str
    count: int = 0
    vehicle_ids: set = field(default_factory=set)

    def increment(self, track_id: int):
        if track_id not in self.vehicle_ids:
            self.vehicle_ids.add(track_id)
            self.count += 1


class LaneCounter:
    def __init__(self, lanes: list[LaneConfig]):
        self.lanes = lanes
        self.counts: dict[str, LaneCount] = {}
        for lane in lanes:
            self.counts[lane.name] = LaneCount(
                lane_name=lane.name,
                direction=lane.direction,
            )

    @staticmethod
    def is_in_lane(point: tuple[int, int], lane: LaneConfig) -> bool:
        if lane.polygon and len(lane.polygon) >= 3:
            pts = np.array(lane.polygon, dtype=np.int32)
            pt = (float(point[0]), float(point[1]))
            return cv2.pointPolygonTest(pts, pt, False) >= 0
        return lane.x_start <= point[0] < lane.x_end

    def update(self, tracks: list) -> dict[str, int]:
        for track in tracks:
            if hasattr(track, "bbox"):
                x1, y1, x2, y2 = track.bbox
                pt = ((x1 + x2) // 2, int(y2))
            else:
                pt = track.centroid
            for lane in self.lanes:
                if self.is_in_lane(pt, lane):
                    self.counts[lane.name].increment(track.track_id)
                    break
        return {name: c.count for name, c in self.counts.items()}

    def get_lane_for_point(self, point: tuple[int, int] | int) -> str | None:
        pt = point if isinstance(point, (tuple, list)) else (int(point), 0)
        for lane in self.lanes:
            if self.is_in_lane(pt, lane):
                return lane.name
        return None

    def reset(self):
        for c in self.counts.values():
            c.count = 0
            c.vehicle_ids.clear()

    @property
    def total(self) -> int:
        return sum(c.count for c in self.counts.values())
