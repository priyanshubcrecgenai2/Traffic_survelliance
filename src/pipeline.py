from __future__ import annotations

import cv2
import time
import signal
import sys
from pathlib import Path

from config import PipelineConfig, get_default_lanes, load_regions_from_json, OUTPUT_DIR
from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.segmentation.instance_segmentor import InstanceSegmentor
from src.tracking.multi_object_tracker import MultiObjectTracker
from src.classification.vehicle_classifier import VehicleClassifier
from src.counting.lane_counter import LaneCounter
from src.counting.direction_counter import DirectionCounter
from src.ocr.anpr_engine import ANPREngine
from src.monitors.pedestrian_cyclist import PedestrianCyclistMonitor
from src.monitors.traffic_rules import TrafficRulesMonitor
from src.storage.event_storage import EventStorage, TrafficEvent
from src.visualizer.evidence import VisualEvidenceGenerator
from src.dashboard.app import Dashboard


class TrafficPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.frame_idx = 0
        self.running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Load custom regions from JSON if available
        loaded_lanes, loaded_zones = [], {}
        if getattr(config, "regions_path", None) and Path(config.regions_path).is_file():
            loaded_lanes, loaded_zones = load_regions_from_json(config.regions_path)
            print(f"[INIT] Loaded {len(loaded_lanes)} lanes and {len(loaded_zones)} zones from {config.regions_path}")

        self.lanes = config.lanes or loaded_lanes
        self.zones = config.zones or loaded_zones

        print(f"[INIT] Loading models on device: {config.device}...")
        self.vehicle_detector = VehicleDetector(
            config.vehicle_model,
            conf=config.confidence,
            iou=config.iou,
            device=config.device,
            regions=self.lanes if getattr(config, "filter_by_regions", True) else [],
            zones=self.zones if getattr(config, "filter_by_regions", True) else {},
        )
        self.plate_detector = PlateDetector(config.plate_model, device=config.device)
        self.segmentor = InstanceSegmentor(config.vehicle_model, device=config.device)
        self.tracker = MultiObjectTracker(max_age=30, iou_threshold=0.3, min_hits=1)
        self.classifier = VehicleClassifier()
        self.storage = EventStorage()
        self.visualizer = VisualEvidenceGenerator(config.output_dir)
        self.ped_cyc_monitor = PedestrianCyclistMonitor(zone_definitions=self.zones)
        self.rules_monitor = TrafficRulesMonitor(self.storage)
        self.lane_counter: LaneCounter | None = None
        self.direction_counter: DirectionCounter | None = None
        self.latest_annotated: np.ndarray | None = None
        self.current_fps: float = 0.0
        self.is_paused: bool = False
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        print("[INIT] Pipeline ready.")

    def set_regions(self, lanes: list[LaneConfig], zones: dict):
        self.lanes = lanes
        self.zones = zones
        self.vehicle_detector.set_regions(lanes, zones)
        self.lane_counter = LaneCounter(lanes)
        self.ped_cyc_monitor.zones = zones or {}

    def get_first_frame(self) -> np.ndarray | None:
        p = Path(self.config.video_path)
        candidates = [str(p), str(p.resolve()), str(Path.cwd() / p), str(Path(__file__).parent.parent / p.name)]
        for cand in candidates:
            if Path(cand).is_file():
                cap = cv2.VideoCapture(cand)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        return frame
        return None

    def _handle_signal(self, sig, frame):
        print("\n[STOP] Interrupt received. Stopping...")
        self.running = False

    def run(self):
        video_path = self.config.video_path
        p = Path(video_path)
        candidates = [str(p), str(p.resolve()), str(Path.cwd() / p), str(Path(__file__).parent.parent / p.name)]
        for cand in candidates:
            if Path(cand).is_file():
                video_path = cand
                break

        print(f"[RUN] Opening video source: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

        if not self.lanes:
            self.lanes = get_default_lanes(width)
            if getattr(self.config, "filter_by_regions", True):
                self.vehicle_detector.set_regions(self.lanes, self.zones)

        self.lane_counter = LaneCounter(self.lanes)
        self.direction_counter = DirectionCounter(height)

        writer = None
        if self.config.save_annotated:
            out_path = str(Path(self.config.output_dir) / "annotated_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        start_time = time.time()
        last_frame_time = time.perf_counter()
        fps_buffer = []

        print("[RUN] Processing started. Press Ctrl+C to stop.")
        while self.running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            self.frame_idx += 1

            # Compute real rolling FPS
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                fps_buffer.append(1.0 / dt)
                if len(fps_buffer) > 20:
                    fps_buffer.pop(0)
                self.current_fps = round(sum(fps_buffer) / len(fps_buffer), 1)

            detections = self.vehicle_detector.detect(frame)
            tracks = self.tracker.update(detections)
            classified = self.classifier.classify_batch(tracks, fps)

            ped_cyc = self.ped_cyc_monitor.update(tracks, self.frame_idx)
            lane_counts = self.lane_counter.update(tracks)
            dir_counts = self.direction_counter.update(tracks)

            # Enforce Wrong-Way and Cycle Zone Traffic Rules
            violations = self.rules_monitor.check_violations(
                tracks, lanes=self.lanes, zones=self.zones, frame_idx=self.frame_idx
            )

            for track in tracks:
                direction = "T2B" if track.velocity[1] > 0 else "B2T"
                bottom_pt = ((track.bbox[0] + track.bbox[2]) // 2, int(track.bbox[3])) if hasattr(track, "bbox") else track.centroid
                lane = self.lane_counter.get_lane_for_point(bottom_pt) or "unknown"

                self.storage.upsert_vehicle(
                    track.track_id, track.class_name, "",
                    lane, direction, 0.0,
                )

            annotated = self.visualizer.annotate_frame(
                frame,
                classified,
                lanes=self.lanes,
                zones=self.zones,
                counts={**lane_counts, **dir_counts},
                violations=violations,
                fps=self.current_fps,
            )

            self.latest_annotated = annotated

            if writer:
                writer.write(annotated)

            if getattr(self.config, "show_live", True):
                try:
                    cv2.imshow("Traffic Intelligence System", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n[STOP] User pressed 'q'. Stopping...")
                        self.running = False
                except cv2.error:
                    self.config.show_live = False

            if self.frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_rate = self.frame_idx / elapsed if elapsed > 0 else 0
                pct = (self.frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"[PROGRESS] Frame {self.frame_idx}/{total_frames} "
                    f"({pct:.1f}%) | {fps_rate:.1f} fps | "
                    f"Vehicles: {len(tracks)} | Lanes: {lane_counts}"
                )

        cap.release()
        if writer:
            writer.release()
        if getattr(self.config, "show_live", True):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        elapsed = time.time() - start_time
        print(f"\n[DONE] Processed {self.frame_idx} frames in {elapsed:.1f}s")
        print(f"[DONE] Avg {self.frame_idx/elapsed:.1f} fps")

        summary = self.storage.get_summary()
        print(f"\n[SUMMARY]")
        print(f"  Total vehicles: {summary['total_vehicles']}")
        print(f"  Total plates: {summary['total_plates']}")
        print(f"  By type: {summary['by_type']}")
        print(f"  By lane: {summary['by_lane']}")
        print(f"  By direction: {summary['by_direction']}")

        self.storage.close()

    def run_with_dashboard(self):
        dashboard = Dashboard(self.storage, pipeline=self, port=self.config.dashboard_port)
        print(f"\n[DASHBOARD] Interactive setup & analytics running at: http://localhost:{self.config.dashboard_port}")
        print(f"[DASHBOARD] Open http://localhost:{self.config.dashboard_port} in your browser to draw regions and start inference.")
        dashboard.run()
