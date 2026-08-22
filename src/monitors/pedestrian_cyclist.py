from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field

from config import PEDESTRIAN_CLASS_IDS, CYCLIST_CLASS_IDS


@dataclass
class ZoneAlert:
    zone_name: str
    alert_type: str  # "pedestrian" or "cyclist"
    count: int
    frame_idx: int
    bboxes: list = field(default_factory=list)


class PedestrianCyclistMonitor:
    def __init__(self, zone_definitions: dict[str, tuple[int, int, int, int]] | None = None):
        self.zones = zone_definitions or {}
        self.pedestrian_count = 0
        self.cyclist_count = 0
        self.pedestrian_ids: set = set()
        self.cyclist_ids: set = set()
        self.alerts: list[ZoneAlert] = []

    def update(self, tracks: list, frame_idx: int = 0) -> dict:
        peds = []
        cyclists = []

        for track in tracks:
            cls = track.class_id
            if cls in PEDESTRIAN_CLASS_IDS:
                self.pedestrian_ids.add(track.track_id)
                peds.append(track)
            elif cls in CYCLIST_CLASS_IDS:
                self.cyclist_ids.add(track.track_id)
                cyclists.append(track)

        self.pedestrian_count = len(self.pedestrian_ids)
        self.cyclist_count = len(self.cyclist_ids)

        if self.zones:
            self._check_zones(peds, cyclists, frame_idx)

        return {
            "pedestrians": self.pedestrian_count,
            "cyclists": self.cyclist_count,
            "active_pedestrians": len(peds),
            "active_cyclists": len(cyclists),
        }

    def _check_zones(self, peds: list, cyclists: list, frame_idx: int):
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            zone_peds = [p for p in peds if self._in_zone(p.centroid, (x1, y1, x2, y2))]
            zone_cyclists = [c for c in cyclists if self._in_zone(c.centroid, (x1, y1, x2, y2))]

            if zone_peds:
                self.alerts.append(ZoneAlert(
                    zone_name=zone_name,
                    alert_type="pedestrian",
                    count=len(zone_peds),
                    frame_idx=frame_idx,
                    bboxes=[p.bbox for p in zone_peds],
                ))
            if zone_cyclists:
                self.alerts.append(ZoneAlert(
                    zone_name=zone_name,
                    alert_type="cyclist",
                    count=len(zone_cyclists),
                    frame_idx=frame_idx,
                    bboxes=[c.bbox for c in zone_cyclists],
                ))

    @staticmethod
    def _in_zone(point: tuple[int, int], zone: tuple[int, int, int, int]) -> bool:
        return zone[0] <= point[0] <= zone[2] and zone[1] <= point[1] <= zone[3]

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, zone_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis

    def reset(self):
        self.pedestrian_count = 0
        self.cyclist_count = 0
        self.pedestrian_ids.clear()
        self.cyclist_ids.clear()
        self.alerts.clear()
