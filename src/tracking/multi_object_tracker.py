from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from src.detection.vehicle_detector import Detection


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
    history: list = field(default_factory=list)

    def __post_init__(self):
        if not self.history:
            self.history = [self.centroid]

    def update(self, detection: Detection):
        prev = self.centroid
        self.bbox = detection.bbox
        self.centroid = detection.centroid
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits += 1
        self.age = 0  # Reset missed frames count on successful detection match
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
        self.frame_count = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        self.frame_count += 1

        if not self.tracks:
            for det in detections:
                self._init_track(det)
            return self.active_tracks

        unmatched_tracks = list(range(len(self.tracks)))
        matched_track_idx = set()
        matched_det_idx = set()

        for t_idx in unmatched_tracks[:]:
            best_iou = 0
            best_d_idx = -1
            for d_idx, det in enumerate(detections):
                if d_idx in matched_det_idx:
                    continue
                iou = self._compute_iou(self.tracks[t_idx].bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx
            if best_iou >= self.iou_threshold and best_d_idx >= 0:
                self.tracks[t_idx].update(detections[best_d_idx])
                matched_track_idx.add(t_idx)
                matched_det_idx.add(best_d_idx)

        for t_idx in range(len(self.tracks)):
            if t_idx not in matched_track_idx:
                self.tracks[t_idx].missed()

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_idx:
                self._init_track(det)

        self.tracks = [t for t in self.tracks if t.age < self.max_age]
        return self.active_tracks

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.is_active(self.min_hits, self.max_age)]

    def _init_track(self, detection: Detection):
        self.tracks.append(Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            centroid=detection.centroid,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
        ))
        self.next_id += 1

    @staticmethod
    def _compute_iou(box1: tuple, box2: tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
