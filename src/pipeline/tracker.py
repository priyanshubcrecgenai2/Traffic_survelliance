from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from src.pipeline.events import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    class_id: int
    class_name: str
    confidence: float
    age: int = 0  # missed frames count
    hits: int = 1
    total_age: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    history: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.history:
            self.history = [self.centroid]

    @property
    def bottom_center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, int(y2))

    def update(self, detection: Detection):
        prev = self.centroid
        self.bbox = detection.bbox
        self.centroid = detection.centroid
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits += 1
        self.age = 0  # Reset missed frames count
        self.total_age += 1
        self.velocity = (
            detection.centroid[0] - prev[0],
            detection.centroid[1] - prev[1],
        )
        self.history.append(self.centroid)

    def missed(self):
        self.age += 1
        self.total_age += 1

    @property
    def active(self) -> bool:
        return self.hits >= 1 and self.age < 30

    def is_active(self, min_hits: int = 1, max_age: int = 30) -> bool:
        return self.hits >= min_hits and self.age < max_age


class MultiObjectTracker:
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3, min_hits: int = 1):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits

    @staticmethod
    def iou(bb_test: tuple[int, int, int, int], bb_gt: tuple[int, int, int, int]) -> float:
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])

        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h

        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_test + area_gt - inter

        return inter / union if union > 0 else 0.0

    def update(self, detections: list[Detection]) -> list[Track]:
        # Associate detections to existing tracks via IoU
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    centroid=det.centroid,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                ))
                self.next_id += 1
            return [t for t in self.tracks if t.is_active(self.min_hits, self.max_age)]

        matched_tracks = set()
        matched_dets = set()

        if len(detections) > 0 and len(self.tracks) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, trk in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = self.iou(trk.bbox, det.bbox)

            # Greedy matching
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_iou < self.iou_threshold:
                    break

                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    iou_matrix[t_idx, d_idx] = -1
                    continue

                self.tracks[t_idx].update(detections[d_idx])
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                iou_matrix[t_idx, :] = -1
                iou_matrix[:, d_idx] = -1

        # Unmatched tracks age
        for t_idx, trk in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                trk.missed()

        # Unmatched detections become new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append(Track(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    centroid=det.centroid,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                ))
                self.next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.age < self.max_age]
        return [t for t in self.tracks if t.is_active(self.min_hits, self.max_age)]
