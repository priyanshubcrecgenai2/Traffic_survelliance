#!/usr/bin/env python3
"""Traffic Intelligence System - Main Entry Point"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, DEFAULT_VIDEO, MODELS_DIR, OUTPUT_DIR, DEVICE
from src.pipeline import TrafficPipeline


def main():
    parser = argparse.ArgumentParser(description="Traffic Intelligence System")
    parser.add_argument(
        "-v", "--video", type=str, default=str(DEFAULT_VIDEO),
        help="Path to input video file",
    )
    parser.add_argument(
        "--vehicle-model", type=str, default=str(MODELS_DIR / "UVH-26-MV-YOLOv11-S.pt"),
        help="Path to vehicle detection YOLO model",
    )
    parser.add_argument(
        "--plate-model", type=str, default=str(MODELS_DIR / "license-plate-finetune-v1l.pt"),
        help="Path to plate detection ONNX model",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default=DEVICE, help="Inference device ('cuda', '0', 'cpu')")
    parser.add_argument("--regions", type=str, default=str(OUTPUT_DIR / "lanes_config.json"), help="Path to lanes_config.json")
    parser.add_argument("--no-filter", dest="filter_regions", action="store_false", default=True, help="Detect vehicles outside regions as well")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--no-annotated", action="store_true", help="Skip saving annotated video")
    parser.add_argument("--show", action="store_true", default=True, help="Display live annotated window (default: True)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable live annotated window")
    parser.add_argument("--dashboard", action="store_true", help="Run web dashboard alongside pipeline")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard port")

    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video,
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        regions_path=args.regions,
        filter_by_regions=args.filter_regions,
        output_dir=args.output,
        save_annotated=not args.no_annotated,
        show_live=args.show,
        dashboard_port=args.port,
    )

    pipeline = TrafficPipeline(config)

    if args.dashboard:
        pipeline.run_with_dashboard()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
