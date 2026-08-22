from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DirectionCount:
    direction: str
    count: int = 0
    vehicle_ids: set = field(default_factory=set)

    def increment(self, track_id: int):
        if track_id not in self.vehicle_ids:
            self.vehicle_ids.add(track_id)
            self.count += 1


class DirectionCounter:
    def __init__(self, frame_height: int, threshold: int = 2):
        self.frame_height = frame_height
        self.threshold = threshold
        self.t2b = DirectionCount(direction="T2B")
        self.b2t = DirectionCount(direction="B2T")
        self.prev_positions: dict[int, tuple[int, int]] = {}

    def update(self, tracks: list) -> dict[str, int]:
        for track in tracks:
            tid = track.track_id
            cy = track.centroid[1]

            if tid in self.prev_positions:
                prev_y = self.prev_positions[tid][1]
                dy = cy - prev_y
                if dy > self.threshold:
                    self.t2b.increment(tid)
                elif dy < -self.threshold:
                    self.b2t.increment(tid)
            self.prev_positions[tid] = track.centroid

        return {"T2B": self.t2b.count, "B2T": self.b2t.count}

    def reset(self):
        self.t2b.count = 0
        self.t2b.vehicle_ids.clear()
        self.b2t.count = 0
        self.b2t.vehicle_ids.clear()
        self.prev_positions.clear()
