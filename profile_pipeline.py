#!/usr/bin/env python3
"""
Pipeline Performance Profiler
Measures detailed latency breakdown of every component in the CV pipeline.
"""

import time
import cv2
import numpy as np
from pathlib import Path

from config import PipelineConfig, DEVICE, MODELS_DIR
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.multi_object_tracker import MultiObjectTracker
from src.classification.vehicle_classifier import VehicleClassifier
from src.counting.lane_counter import LaneCounter
from src.counting.direction_counter import DirectionCounter
from src.monitors.traffic_rules import TrafficRulesMonitor
from src.storage.event_storage import EventStorage
from src.visualizer.evidence import VisualEvidenceGenerator
from config import get_default_lanes, load_regions_from_json

def profile(num_frames: int = 200):
    video_path = "traffic_2.mp4"
    if not Path(video_path).is_file():
        video_path = "traffic_1.mp4"
        
    print(f"[PROFILE] Opening {video_path} on device {DEVICE} for {num_frames} frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Load regions
    lanes, zones = [], {}
    if Path("output/lanes_config.json").is_file():
        lanes, zones = load_regions_from_json("output/lanes_config.json")
    if not lanes:
        lanes = get_default_lanes(width)

    detector = VehicleDetector(str(MODELS_DIR / "UVH-26-MV-YOLOv11-S.pt"), device=DEVICE, regions=lanes, zones=zones)
    tracker = MultiObjectTracker(max_age=30, iou_threshold=0.3, min_hits=1)
    classifier = VehicleClassifier()
    lane_counter = LaneCounter(lanes)
    dir_counter = DirectionCounter(height)
    storage = EventStorage("output/test_profile.db")
    rules_monitor = TrafficRulesMonitor(storage)
    visualizer = VisualEvidenceGenerator("output")

    times = {
        "decode": [],
        "yolo_detect": [],
        "tracking": [],
        "classification": [],
        "rules_and_lanes": [],
        "db_write": [],
        "rendering": [],
        "total_frame": []
    }

    # Warmup 10 frames
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            detector.detect(frame)

    print(f"[PROFILE] Profiling {num_frames} frames...")
    frame_idx = 0
    t_start = time.perf_counter()

    while frame_idx < num_frames:
        t0 = time.perf_counter()
        
        # 1. Decode
        t_d0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t_d1 = time.perf_counter()
        times["decode"].append((t_d1 - t_d0) * 1000)

        # 2. YOLO Detect
        t_y0 = time.perf_counter()
        detections = detector.detect(frame)
        t_y1 = time.perf_counter()
        times["yolo_detect"].append((t_y1 - t_y0) * 1000)

        # 3. Tracking
        t_t0 = time.perf_counter()
        tracks = tracker.update(detections)
        t_t1 = time.perf_counter()
        times["tracking"].append((t_t1 - t_t0) * 1000)

        # 4. Classification
        t_c0 = time.perf_counter()
        classified = classifier.classify_batch(tracks, fps)
        t_c1 = time.perf_counter()
        times["classification"].append((t_c1 - t_c0) * 1000)

        # 5. Rules & Counting
        t_r0 = time.perf_counter()
        lane_counts = lane_counter.update(tracks)
        dir_counts = dir_counter.update(tracks)
        violations = rules_monitor.check_violations(tracks, lanes=lanes, zones=zones, frame_idx=frame_idx)
        t_r1 = time.perf_counter()
        times["rules_and_lanes"].append((t_r1 - t_r0) * 1000)

        # 6. Database Writes
        t_db0 = time.perf_counter()
        for track in tracks:
            direction = "T2B" if track.velocity[1] > 0 else "B2T"
            bottom_pt = ((track.bbox[0] + track.bbox[2]) // 2, int(track.bbox[3]))
            lane = lane_counter.get_lane_for_point(bottom_pt) or "unknown"
            storage.upsert_vehicle(track.track_id, track.class_name, "", lane, direction, 0.0)
        t_db1 = time.perf_counter()
        times["db_write"].append((t_db1 - t_db0) * 1000)

        # 7. Rendering
        t_rend0 = time.perf_counter()
        annotated = visualizer.annotate_frame(
            frame, classified, lanes=lanes, zones=zones,
            counts={**lane_counts, **dir_counts}, violations=violations
        )
        t_rend1 = time.perf_counter()
        times["rendering"].append((t_rend1 - t_rend0) * 1000)

        t_end = time.perf_counter()
        times["total_frame"].append((t_end - t0) * 1000)
        frame_idx += 1

    cap.release()
    storage.close()
    t_total = time.perf_counter() - t_start

    print("\n" + "="*60)
    print("PERFORMANCE PROFILING BREAKDOWN (ms per frame)")
    print("="*60)
    for k, v in times.items():
        avg = np.mean(v)
        p95 = np.percentile(v, 95)
        print(f"{k:<18}: Mean: {avg:6.2f} ms | P95: {p95:6.2f} ms | {avg/np.mean(times['total_frame'])*100:4.1f}%")
    print("-" * 60)
    print(f"Total Measured Real FPS: {frame_idx / t_total:.1f} FPS")
    print(f"Theoretical Max CV FPS (excluding decode/render): {1000 / (np.mean(times['yolo_detect']) + np.mean(times['tracking']) + np.mean(times['rules_and_lanes'])):.1f} FPS")
    print("="*60)

if __name__ == "__main__":
    profile(150)
