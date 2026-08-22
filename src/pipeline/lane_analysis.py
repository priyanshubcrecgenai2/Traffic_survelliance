from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from config import LaneConfig
from src.pipeline.tracker import Track


@dataclass
class LaneStats:
    name: str
    direction: str
    count: int = 0
    t2b_count: int = 0
    b2t_count: int = 0
    vehicle_ids: set[int] = field(default_factory=set)

    def register_vehicle(self, track_id: int, direction: str):
        if track_id not in self.vehicle_ids:
            self.vehicle_ids.add(track_id)
            self.count += 1
            if direction == "T2B":
                self.t2b_count += 1
            elif direction == "B2T":
                self.b2t_count += 1


class LaneAnalyzer:
    def __init__(self, lanes: list[LaneConfig] | None = None, frame_height: int = 1080):
        self.lanes = lanes or []
        self.frame_height = frame_height
        self.lane_stats: dict[str, LaneStats] = {
            lane.name: LaneStats(name=lane.name, direction=lane.direction)
            for lane in self.lanes
        }
        self.dir_counts: dict[str, int] = {"T2B": 0, "B2T": 0}
        self.dir_seen_ids: set[tuple[int, str]] = set()

    def set_lanes(self, lanes: list[LaneConfig]):
        self.lanes = lanes
        self.lane_stats = {
            lane.name: LaneStats(name=lane.name, direction=lane.direction)
            for lane in lanes
        }

    @staticmethod
    def is_in_lane(point: tuple[int, int], lane: LaneConfig) -> bool:
        if lane.polygon and len(lane.polygon) >= 3:
            pts = np.array(lane.polygon, dtype=np.int32)
            pt = (float(point[0]), float(point[1]))
            return cv2.pointPolygonTest(pts, pt, False) >= 0
        return lane.x_start <= point[0] < lane.x_end

    def get_lane_for_point(self, point: tuple[int, int]) -> str | None:
        for lane in self.lanes:
            if self.is_in_lane(point, lane):
                return lane.name
        return None

    def determine_direction(self, track: Track) -> str:
        """Determines movement direction (T2B or B2T) based on trajectory history."""
        if len(track.history) >= 2:
            dy = track.history[-1][1] - track.history[0][1]
            if len(track.history) >= 5:
                dy_recent = track.history[-1][1] - track.history[-5][1]
                if abs(dy_recent) > 2.0:
                    return "T2B" if dy_recent > 0 else "B2T"
            if abs(dy) > 2.0:
                return "T2B" if dy > 0 else "B2T"
        
        # Fallback to instantaneous velocity
        return "T2B" if track.velocity[1] > 0 else "B2T"

    def update(self, tracks: list[Track]) -> dict:
        """Updates lane statistics and directional counts for active tracks."""
        for track in tracks:
            contact_pt = track.bottom_center
            direction = self.determine_direction(track)

            # Update directional totals
            dir_key = (track.track_id, direction)
            if dir_key not in self.dir_seen_ids:
                self.dir_seen_ids.add(dir_key)
                self.dir_counts[direction] = self.dir_counts.get(direction, 0) + 1

            # Update lane totals
            for lane in self.lanes:
                if self.is_in_lane(contact_pt, lane):
                    if lane.name in self.lane_stats:
                        self.lane_stats[lane.name].register_vehicle(track.track_id, direction)
                    break

        return {
            "lane_counts": {name: s.count for name, s in self.lane_stats.items()},
            "dir_counts": dict(self.dir_counts),
        }
