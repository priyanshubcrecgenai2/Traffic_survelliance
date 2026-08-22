from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from src.pipeline.tracker import Track
from src.pipeline.events import TrafficViolation


class VisualEvidence:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.annotated_dir = self.output_dir / "annotated_frames"
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lanes: list | None = None,
        zones: dict | list | None = None,
        counts: dict | None = None,
        violations: dict[int, TrafficViolation] | None = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        vis = frame.copy()
        violations = violations or {}

        # 1. Draw Lane Overlays (Exact Polygons)
        if lanes:
            overlay = vis.copy()
            lane_colors = [
                (0, 255, 0), (255, 165, 0), (255, 0, 255),
                (0, 255, 255), (255, 255, 0), (128, 255, 0)
            ]
            for idx, lane in enumerate(lanes):
                poly = getattr(lane, "polygon", [])
                c = lane_colors[idx % len(lane_colors)]
                if poly and len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], c)
                    cv2.polylines(vis, [pts], True, c, 2)
                    cx = int(np.mean([p[0] for p in poly]))
                    cy = int(np.mean([p[1] for p in poly]))
                    name = getattr(lane, "name", f"lane_{idx+1}")
                    dir_str = getattr(lane, "direction", "")
                    label = f"{name} ({dir_str})" if dir_str else name
                    cv2.putText(vis, label, (cx - 35, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)

        # 2. Draw Monitoring Zones (Cycle / Pedestrian Zones)
        if zones:
            items = zones.items() if isinstance(zones, dict) else enumerate(zones)
            for z_name, z_val in items:
                name_str = str(z_name) if isinstance(zones, dict) else f"zone_{z_name+1}"
                if isinstance(z_val, (list, tuple)):
                    if len(z_val) == 4 and all(isinstance(v, (int, float)) for v in z_val):
                        zx1, zy1, zx2, zy2 = map(int, z_val)
                        cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
                        cv2.putText(vis, name_str, (zx1 + 5, zy1 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    elif len(z_val) >= 3:
                        pts = np.array(z_val, dtype=np.int32)
                        cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
                        cx = int(np.mean([p[0] for p in z_val]))
                        cy = int(np.mean([p[1] for p in z_val]))
                        cv2.putText(vis, name_str, (cx - 25, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 3. Draw Tracked Vehicles & Bounding Boxes
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255),
        ]

        for track in tracks:
            is_violating = track.track_id in violations
            viol_info = violations.get(track.track_id)

            color = (0, 0, 255) if is_violating else colors[track.track_id % len(colors)]
            thickness = 3 if is_violating else 2
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            cls_name = getattr(track, "class_name", "")
            if is_violating and viol_info:
                viol_tag = "WRONG WAY" if viol_info.violation_type == "WRONG_WAY" else "ZONE INTRUSION"
                label = f"ID:{track.track_id} {cls_name} [{viol_tag}]"
            else:
                label = f"ID:{track.track_id} {cls_name}".strip()

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            text_color = (255, 255, 255) if is_violating else (0, 0, 0)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Draw trajectory trail
            if hasattr(track, "history") and len(track.history) > 1:
                pts = np.array(track.history[-25:], dtype=np.int32)
                cv2.polylines(vis, [pts], False, color, 2)

        # 4. Top Right Alert Banner
        if violations:
            cv2.rectangle(vis, (vis.shape[1] - 380, 10), (vis.shape[1] - 10, 50), (0, 0, 220), -1)
            cv2.putText(vis, f"ALERT: {len(violations)} RULE VIOLATION(S)",
                        (vis.shape[1] - 370, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 5. Top Left Telemetry HUD (FPS & Live Counts)
        if fps > 0:
            cv2.rectangle(vis, (10, 10), (160, 45), (15, 23, 42), -1)
            cv2.rectangle(vis, (10, 10), (160, 45), (56, 189, 248), 1)
            cv2.putText(vis, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)

        if counts:
            y_offset = 75 if fps > 0 else 30
            for key, val in counts.items():
                text = f"{key}: {val}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(vis, (10, y_offset - th - 4), (10 + tw + 10, y_offset + 4), (15, 23, 42), -1)
                cv2.putText(vis, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 26

        return vis
