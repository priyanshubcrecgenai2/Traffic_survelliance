from __future__ import annotations

import cv2
import time
import numpy as np
from config import LaneConfig
from src.pipeline.tracker import Track
from src.pipeline.events import TrafficViolation


ALLOWED_CYCLE_KEYWORDS = (
    "bicycle", "bike", "cycle", "motorcycle", "motorbike", "motor_bike",
    "scooter", "scooty", "two_wheeler", "two-wheeler", "2-wheeler",
    "pedestrian", "person", "cyclist", "rider", "walker"
)

UNAUTHORIZED_CYCLE_KEYWORDS = (
    "car", "truck", "bus", "auto", "rickshaw", "auto_rickshaw", "van",
    "suv", "lorry", "tractor", "jeep", "pickup", "heavy", "vehicle"
)


class TrafficRuleEngine:
    def __init__(self, db_storage=None):
        self.db_storage = db_storage
        self.active_violations: dict[int, TrafficViolation] = {}
        self.logged_violations: set[tuple[int, str]] = set()
        self.streaks: dict[tuple[int, str], int] = {}  # (track_id, rule) -> consecutive frames

    def _in_polygon(self, point: tuple[int, int], polygon: list[tuple[int, int]] | list[list[int]]) -> bool:
        if len(polygon) < 3:
            return False
        poly_np = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(poly_np, (float(point[0]), float(point[1])), False) >= 0

    def evaluate(
        self,
        tracks: list[Track],
        lanes: list[LaneConfig] | None = None,
        zones: dict | list | None = None,
        frame_idx: int = 0
    ) -> dict[int, TrafficViolation]:
        current_active = {}
        seen_keys = set()

        for track in tracks:
            cls_name = getattr(track, "class_name", "").lower().strip()
            x1, y1, x2, y2 = track.bbox
            bottom_center = track.bottom_center

            # -----------------------------------------------------------------
            # 1. Rule Check: Direction Enforcement (Wrong-Way Driving)
            # -----------------------------------------------------------------
            if lanes and len(track.history) >= 8 and track.hits >= 6:
                h = track.history
                dy_recent = h[-1][1] - h[-7][1]
                dy_total = h[-1][1] - h[0][1]

                for lane in lanes:
                    lane_poly = getattr(lane, "polygon", [])
                    if not lane_poly:
                        continue

                    if self._in_polygon(bottom_center, lane_poly):
                        expected_dir = getattr(lane, "direction", "").upper()
                        is_wrong_way = False
                        moving_dir = None

                        if expected_dir == "B2T":
                            # Expected UP (y decreases). If moving DOWN significantly:
                            if dy_recent >= 12.0 and dy_total >= 18.0:
                                is_wrong_way = True
                                moving_dir = "T2B"
                        elif expected_dir == "T2B":
                            # Expected DOWN (y increases). If moving UP significantly:
                            if dy_recent <= -12.0 and dy_total <= -18.0:
                                is_wrong_way = True
                                moving_dir = "B2T"

                        streak_key = (track.track_id, "WRONG_WAY")
                        seen_keys.add(streak_key)

                        if is_wrong_way and moving_dir:
                            self.streaks[streak_key] = self.streaks.get(streak_key, 0) + 1
                            if self.streaks[streak_key] >= 4:
                                v = TrafficViolation(
                                    track_id=track.track_id,
                                    violation_type="WRONG_WAY",
                                    description=f"Wrong-Way: moving {moving_dir} in {lane.name} ({expected_dir})",
                                    vehicle_type=cls_name or "vehicle",
                                    lane=lane.name,
                                    timestamp=time.time(),
                                    frame_idx=frame_idx,
                                )
                                current_active[track.track_id] = v
                                self._log_violation(v, bbox=f"{x1},{y1},{x2},{y2}")
                        else:
                            self.streaks[streak_key] = 0

            # -----------------------------------------------------------------
            # 2. Rule Check: Cycle / Pedestrian Zone Intrusion
            # -----------------------------------------------------------------
            if zones:
                is_allowed = any(kw in cls_name for kw in ALLOWED_CYCLE_KEYWORDS)
                is_unauthorized = any(kw in cls_name for kw in UNAUTHORIZED_CYCLE_KEYWORDS) or (not is_allowed and cls_name not in ("", "unknown"))

                if is_unauthorized and not is_allowed:
                    items = zones.items() if isinstance(zones, dict) else enumerate(zones)
                    for z_name, z_val in items:
                        if isinstance(z_val, (list, tuple)) and len(z_val) >= 3:
                            streak_key = (track.track_id, f"ZONE_{z_name}")
                            seen_keys.add(streak_key)

                            if self._in_polygon(bottom_center, z_val):
                                self.streaks[streak_key] = self.streaks.get(streak_key, 0) + 1
                                if self.streaks[streak_key] >= 3:
                                    v = TrafficViolation(
                                        track_id=track.track_id,
                                        violation_type="CYCLE_ZONE_INTRUSION",
                                        description=f"Cycle Zone Intrusion: {cls_name} entered {z_name}",
                                        vehicle_type=cls_name or "vehicle",
                                        lane=str(z_name),
                                        timestamp=time.time(),
                                        frame_idx=frame_idx,
                                    )
                                    current_active[track.track_id] = v
                                    self._log_violation(v, bbox=f"{x1},{y1},{x2},{y2}")
                            else:
                                self.streaks[streak_key] = 0

        # Clean up expired streaks
        for k in list(self.streaks.keys()):
            if k not in seen_keys:
                self.streaks.pop(k, None)

        self.active_violations = current_active
        return current_active

    def _log_violation(self, violation: TrafficViolation, bbox: str = ""):
        key = (violation.track_id, violation.violation_type)
        if key not in self.logged_violations:
            self.logged_violations.add(key)
            if self.db_storage:
                self.db_storage.log_violation_event(
                    timestamp=violation.timestamp,
                    frame_idx=violation.frame_idx,
                    violation_type=violation.violation_type,
                    track_id=violation.track_id,
                    vehicle_type=violation.vehicle_type,
                    lane=violation.lane,
                    description=violation.description,
                    bbox=bbox,
                )
