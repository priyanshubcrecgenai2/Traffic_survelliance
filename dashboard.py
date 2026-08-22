#!/usr/bin/env python3
"""
Traffic Intelligence Interactive Web Dashboard Launcher
Allows drawing polygon regions for Lane 1, Lane 2, and Cycle / Pedestrian Zone on the first frame,
then initiates live YOLO NVIDIA GPU inference and streams live analytics.

Usage:
    python dashboard.py -v traffic_2.mp4
    python dashboard.py -v traffic_2.mp4 --port 5000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, DEFAULT_VIDEO, MODELS_DIR, OUTPUT_DIR, DEVICE
from src.pipeline import TrafficPipeline


def main():
    parser = argparse.ArgumentParser(description="Traffic Intelligence Web Dashboard & Region Setup")
    parser.add_argument("-v", "--video", type=str, default=str(DEFAULT_VIDEO), help="Path to input video")
    parser.add_argument("--vehicle-model", type=str, default=str(MODELS_DIR / "UVH-26-MV-YOLOv11-S.pt"), help="Vehicle model")
    parser.add_argument("--plate-model", type=str, default=str(MODELS_DIR / "license-plate-finetune-v1l.pt"), help="Plate model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default=DEVICE, help="Inference device (e.g. 'cuda', '0', 'cpu')")
    parser.add_argument("--regions", type=str, default=str(OUTPUT_DIR / "lanes_config.json"), help="Path to lanes_config.json")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard web port (default: 5000)")
    parser.add_argument("--no-annotated", action="store_true", help="Skip saving annotated video to disk")

    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video,
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        regions_path=args.regions,
        output_dir=args.output,
        save_annotated=not args.no_annotated,
        show_live=False,  # Stream directly inside web dashboard
        dashboard_port=args.port,
    )

    pipeline = TrafficPipeline(config)
    pipeline.run_with_dashboard()


if __name__ == "__main__":
    main()
