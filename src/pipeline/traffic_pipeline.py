from __future__ import annotations

import cv2
import time
import signal
import numpy as np
from pathlib import Path

from config import PipelineConfig, LaneConfig, load_regions_from_json, get_default_lanes
from src.pipeline.detector import VehicleDetector
from src.pipeline.tracker import MultiObjectTracker
from src.pipeline.lane_analysis import LaneAnalyzer
from src.pipeline.traffic_rules import TrafficRuleEngine
from src.storage.database import EventDatabase
from src.visualization.evidence import VisualEvidence


class TrafficPipeline:
    """
    Central pipeline orchestrating end-to-end traffic surveillance intelligence:
    Video Input -> YOLO Detection -> Tracking -> Lane & Direction -> Rule Enforcement -> DB Store -> Visualization
    """
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.running = False
        self.is_paused = False
        self.frame_idx = 0
        self.current_fps = 0.0
        self.latest_annotated: np.ndarray | None = None

        # Load persisted region configuration if available
        loaded_lanes, loaded_zones = [], {}
        if self.config.regions_path and Path(self.config.regions_path).is_file():
            loaded_lanes, loaded_zones = load_regions_from_json(self.config.regions_path)

        self.lanes = self.config.lanes or loaded_lanes
        self.zones = self.config.zones or loaded_zones

        # Initialize Computer Vision Components
        print(f"[PIPELINE INIT] Initializing YOLO on device: {self.config.device}...")
        self.detector = VehicleDetector(
            self.config.vehicle_model,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.config.device,
            regions=self.lanes if getattr(self.config, "filter_by_regions", True) else [],
            zones=self.zones if getattr(self.config, "filter_by_regions", True) else {},
        )
        self.tracker = MultiObjectTracker(max_age=30, iou_threshold=0.3, min_hits=1)
        self.lane_analyzer = LaneAnalyzer(self.lanes)
        self.database = EventDatabase()
        self.rule_engine = TrafficRuleEngine(self.database)
        self.visualizer = VisualEvidence(self.config.output_dir)

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        print("[PIPELINE INIT] Ready.")

    def set_regions(self, lanes: list[LaneConfig], zones: dict):
        """Updates lane and zone polygon regions dynamically."""
        self.lanes = lanes
        self.zones = zones
        self.detector.set_regions(lanes, zones)
        self.lane_analyzer.set_lanes(lanes)

    def get_first_frame(self) -> np.ndarray | None:
        """Retrieves the first frame from the video source for interactive canvas setup."""
        video_path = self._resolve_video_path(self.config.video_path)
        if not video_path:
            return None
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame
        return None

    def _resolve_video_path(self, path_str: str) -> str | None:
        p = Path(path_str)
        candidates = [
            str(p),
            str(p.resolve()),
            str(Path.cwd() / p),
            str(Path.cwd() / p.name),
            "traffic_2.mp4",
            "traffic_1.mp4",
        ]
        for cand in candidates:
            if Path(cand).is_file():
                return cand
        return None

    def run(self):
        """Executes the core computer vision processing loop."""
        video_path = self._resolve_video_path(self.config.video_path)
        if not video_path:
            print(f"[PIPELINE ERROR] Video not found: {self.config.video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[PIPELINE ERROR] Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[PIPELINE RUN] {width}x{height} @ {fps:.1f}fps ({total_frames} frames)")

        # Apply default regions if none defined
        if not self.lanes:
            self.lanes = get_default_lanes(width)
            self.detector.set_regions(self.lanes, self.zones)
            self.lane_analyzer.set_lanes(self.lanes)

        writer = None
        if self.config.save_annotated:
            out_path = str(Path(self.config.output_dir) / "annotated_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        self.running = True
        last_frame_time = time.perf_counter()
        fps_buffer = []

        print("[PIPELINE RUN] Processing started.")
        while self.running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            self.frame_idx += 1

            # 1. Rolling Measured FPS Calculation
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                fps_buffer.append(1.0 / dt)
                if len(fps_buffer) > 20:
                    fps_buffer.pop(0)
                self.current_fps = round(sum(fps_buffer) / len(fps_buffer), 1)

            # 2. Vehicle Detection (YOLOv11 Tensor Core FP16)
            detections = self.detector.detect(frame)

            # 3. Multi-Object Tracking (Persistent Track IDs)
            tracks = self.tracker.update(detections)

            # 4. Lane & Direction Analysis (T2B vs B2T)
            counts = self.lane_analyzer.update(tracks)
            lane_counts = counts.get("lane_counts", {})
            dir_counts = counts.get("dir_counts", {})

            # 5. Traffic Rules & Violation Enforcement
            violations = self.rule_engine.evaluate(
                tracks, lanes=self.lanes, zones=self.zones, frame_idx=self.frame_idx
            )

            # 6. Database Persistence (Async Non-Blocking Queue)
            for track in tracks:
                direction = self.lane_analyzer.determine_direction(track)
                lane = self.lane_analyzer.get_lane_for_point(track.bottom_center) or "unknown"
                self.database.upsert_vehicle(
                    track.track_id, track.class_name, "",
                    lane, direction, 0.0,
                )

            # 7. Visual HUD & Evidence Rendering
            annotated = self.visualizer.annotate_frame(
                frame,
                tracks,
                lanes=self.lanes,
                zones=self.zones,
                counts={**lane_counts, **dir_counts},
                violations=violations,
                fps=self.current_fps,
            )

            self.latest_annotated = annotated
            if writer:
                writer.write(annotated)

        cap.release()
        if writer:
            writer.release()
        self.running = False
        print("[PIPELINE STOP] Video processing completed.")

    def stop(self):
        self.running = False
        if hasattr(self, "database"):
            self.database.close()
