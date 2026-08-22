#!/usr/bin/env python3
"""
Traffic Surveillance & Intelligence System — Application Entry Point
Launches the TrafficPipeline and web application server.

Usage:
    python dashboard.py -v traffic_2.mp4
    python dashboard.py -v traffic_2.mp4 --port 5000
"""

import sys
import argparse
from pathlib import Path

from config import PipelineConfig, DEFAULT_VIDEO, MODELS_DIR, OUTPUT_DIR, DEVICE
from src.pipeline.traffic_pipeline import TrafficPipeline
from src.web.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Traffic Surveillance & Intelligence System")
    parser.add_argument("-v", "--video", type=str, default=str(DEFAULT_VIDEO), help="Path to input video file")
    parser.add_argument("--vehicle-model", type=str, default=str(MODELS_DIR / "UVH-26-MV-YOLOv11-S.pt"), help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--device", type=str, default=DEVICE, help="Inference device ('cuda', '0', 'cpu')")
    parser.add_argument("--port", type=int, default=5000, help="Web dashboard port (default: 5000)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--no-annotated", action="store_true", help="Skip saving annotated video")

    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video,
        vehicle_model=args.vehicle_model,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        output_dir=args.output,
        save_annotated=not args.no_annotated,
        dashboard_port=args.port,
    )

    pipeline = TrafficPipeline(config)
    app = create_app(pipeline=pipeline, port=args.port)

    print(f"\n=======================================================")
    print(f"🚦 TRAFFIC INTELLIGENCE PLATFORM READY")
    print(f"   • Live Monitoring & Region Setup : http://localhost:{args.port}")
    print(f"   • Operations Analytics Center    : http://localhost:{args.port}/analytics")
    print(f"=======================================================\n")

    try:
        from werkzeug.serving import make_server
        server = make_server("0.0.0.0", args.port, app, threaded=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping Traffic Intelligence System...")
        pipeline.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
