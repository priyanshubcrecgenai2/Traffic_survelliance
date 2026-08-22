from __future__ import annotations

import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response

from config import LaneConfig, OUTPUT_DIR, REGIONS_CONFIG_PATH
from src.storage.database import EventDatabase


def create_app(pipeline=None, storage: EventDatabase | None = None, port: int = 5000) -> Flask:
    template_dir = Path(__file__).parent.parent.parent / "templates"
    static_dir = Path(__file__).parent.parent.parent / "static"
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    inference_thread: list[threading.Thread | None] = [None]

    db = storage or (pipeline.database if pipeline else EventDatabase())

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/analytics")
    def analytics():
        return render_template("analytics.html")

    @app.route("/api/first_frame")
    def api_first_frame():
        frame = None
        if pipeline:
            frame = pipeline.get_first_frame()
        if frame is None:
            video_name = pipeline.config.video_path if pipeline else "traffic_2.mp4"
            candidates = [
                video_name,
                str(Path(video_name).resolve()),
                str(Path.cwd() / video_name),
                str(Path.cwd() / Path(video_name).name),
                "traffic_2.mp4",
                "traffic_1.mp4",
            ]
            for cand in candidates:
                if Path(cand).is_file():
                    cap = cv2.VideoCapture(cand)
                    if cap.isOpened():
                        ret, f = cap.read()
                        cap.release()
                        if ret and f is not None:
                            frame = f
                            break
        if frame is None:
            frame = 20 * np.ones((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "No Frame Available", (400, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        resp = Response(buf.tobytes(), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/api/get_regions")
    def api_get_regions():
        reg_path = Path(REGIONS_CONFIG_PATH)
        if not reg_path.is_file():
            reg_path = Path(OUTPUT_DIR) / "lanes_config.json"

        if reg_path.is_file():
            try:
                with open(reg_path, "r") as f:
                    return jsonify(json.load(f))
            except Exception:
                pass
        if pipeline and pipeline.lanes:
            lanes_data = [
                {"name": l.name, "direction": l.direction, "polygon": l.polygon}
                for l in pipeline.lanes
            ]
            return jsonify({"lanes": lanes_data, "zones": pipeline.zones or {}})
        return jsonify({"lanes": [], "zones": {}})

    @app.route("/api/start_inference", methods=["POST"])
    def api_start_inference():
        data = request.get_json() or {}
        lanes_data = data.get("lanes", [])
        zones_data = data.get("zones", {})

        # Persist regions to config file
        out_file = Path(REGIONS_CONFIG_PATH)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump({"lanes": lanes_data, "zones": zones_data}, f, indent=2)

        lanes_objs = [
            LaneConfig(
                name=l["name"],
                direction=l.get("direction", "T2B"),
                polygon=[(int(p[0]), int(p[1])) for p in l.get("polygon", [])],
            )
            for l in lanes_data
        ]

        if pipeline:
            pipeline.set_regions(lanes_objs, zones_data)
            if inference_thread[0] is None or not inference_thread[0].is_alive():
                pipeline.running = True
                pipeline.is_paused = False
                t = threading.Thread(target=pipeline.run, daemon=True)
                inference_thread[0] = t
                t.start()

        return jsonify({"status": "ok", "lanes_count": len(lanes_objs)})

    @app.route("/api/pause", methods=["POST"])
    def api_pause():
        if pipeline:
            pipeline.is_paused = not pipeline.is_paused
            return jsonify({"status": "ok", "is_paused": pipeline.is_paused})
        return jsonify({"status": "error"})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        if pipeline:
            pipeline.running = False
            return jsonify({"status": "ok"})
        return jsonify({"status": "error"})

    @app.route("/api/summary")
    def api_summary():
        summary = db.get_summary()
        if pipeline:
            summary["frame_idx"] = pipeline.frame_idx
            summary["is_running"] = pipeline.running
            summary["is_paused"] = pipeline.is_paused
            summary["fps"] = getattr(pipeline, "current_fps", 0.0)
        return jsonify(summary)

    @app.route("/api/violations")
    def api_violations():
        return jsonify(db.get_violations(30))

    @app.route("/api/analytics")
    def api_analytics():
        time_range = request.args.get("time_range", "live")
        payload = db.get_analytics_payload(time_range=time_range)
        payload["system_status"] = {
            "is_running": pipeline.running if pipeline else False,
            "is_paused": pipeline.is_paused if pipeline else False,
            "fps": getattr(pipeline, "current_fps", 0.0) if pipeline else 0.0,
            "frame_idx": getattr(pipeline, "frame_idx", 0) if pipeline else 0,
            "video_name": Path(pipeline.config.video_path).name if pipeline else "traffic_2.mp4",
        }
        return jsonify(payload)

    @app.route("/video_feed")
    def video_feed():
        def generate():
            init_frame = None
            if pipeline:
                init_frame = pipeline.latest_annotated or pipeline.get_first_frame()
            if init_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", init_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

            while True:
                if pipeline and pipeline.latest_annotated is not None:
                    frame = pipeline.latest_annotated
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(0.033)  # ~30 FPS
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app
