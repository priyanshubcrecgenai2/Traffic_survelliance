#!/usr/bin/env python3
"""
Interactive Polygon Lane and Region Drawer Tool.
Allows drawing angled/perspective 4-point polygon lanes and monitoring regions directly on video frames.

Usage:
    python scripts/draw_regions.py -v traffic_2.mp4
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_VIDEO, REGIONS_CONFIG_PATH


class PolygonDrawer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise ValueError(f"Cannot read frame from video: {video_path}")
        self.cap.release()

        self.orig_frame = frame
        self.h, self.w = frame.shape[:2]

        self.mode = "lane"  # "lane" or "zone"
        self.lanes: list[dict] = []
        self.zones: dict[str, list[list[int]]] = {}
        self.current_points: list[tuple[int, int]] = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))

    def complete_current_shape(self):
        if len(self.current_points) >= 3:
            if self.mode == "lane":
                lane_num = len(self.lanes) + 1
                direction = "T2B" if lane_num % 2 == 0 else "B2T"
                self.lanes.append({
                    "name": f"lane_{lane_num}",
                    "direction": direction,
                    "polygon": [list(p) for p in self.current_points],
                })
                print(f"[LANE ADDED] lane_{lane_num} with {len(self.current_points)} vertices ({direction})")
            elif self.mode == "zone":
                zone_num = len(self.zones) + 1
                z_name = f"zone_{zone_num}"
                self.zones[z_name] = [list(p) for p in self.current_points]
                print(f"[ZONE ADDED] {z_name} with {len(self.current_points)} vertices")
            self.current_points.clear()
        else:
            print("[WARN] Need at least 3 points to complete a polygon region")

    def render(self) -> np.ndarray:
        vis = self.orig_frame.copy()
        overlay = vis.copy()

        colors = [
            (0, 255, 0), (255, 165, 0), (255, 0, 255),
            (0, 255, 255), (255, 255, 0), (128, 255, 0)
        ]
        for i, lane in enumerate(self.lanes):
            pts = np.array(lane["polygon"], dtype=np.int32)
            c = colors[i % len(colors)]
            cv2.fillPoly(overlay, [pts], c)
            cv2.polylines(vis, [pts], True, c, 2)
            cx = int(np.mean([p[0] for p in lane["polygon"]]))
            cy = int(np.mean([p[1] for p in lane["polygon"]]))
            label = f"{lane['name']} ({lane['direction']})"
            cv2.putText(vis, label, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for z_name, z_pts in self.zones.items():
            pts = np.array(z_pts, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
            cx = int(np.mean([p[0] for p in z_pts]))
            cy = int(np.mean([p[1] for p in z_pts]))
            cv2.putText(vis, z_name, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        if self.current_points:
            active_c = (0, 255, 0) if self.mode == "lane" else (0, 255, 255)
            for pt in self.current_points:
                cv2.circle(vis, pt, 5, active_c, -1)
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(vis, [pts], False, active_c, 2)

        # Top HUD instructions
        hud = np.zeros((70, self.w, 3), dtype=np.uint8)
        mode_str = f"MODE: {'POLYGON LANE' if self.mode == 'lane' else 'MONITOR ZONE'}"
        pts_str = f"Vertices: {len(self.current_points)}"
        cv2.putText(hud, f"{mode_str} | {pts_str} | [L]ane | [Z]one | [F]inish Shape | [U]ndo | [C]lear | [S]ave | [Q]uit",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(hud, "Click 4 corner points around each lane/region -> Press 'F' or ENTER to finish",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        return np.vstack([hud, vis])

    def save_config(self) -> str:
        config_data = {
            "lanes": self.lanes,
            "zones": self.zones,
        }

        out_file = Path(REGIONS_CONFIG_PATH)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"\n[SAVED] Region configuration successfully saved to {out_file}\n")
        return str(out_file)

    def run(self):
        win_name = "Perspective Polygon Lane & Zone Configurator"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        print("\n=== Perspective Polygon Lane & Region Configurator ===")
        print("  - Click points to mark corners of a lane or region.")
        print("  - Press 'f' or ENTER: Finish current polygon shape.")
        print("  - Press 'l': Switch to Lane Polygon mode.")
        print("  - Press 'z': Switch to Monitoring Zone mode.")
        print("  - Press 'u': Undo last vertex point.")
        print("  - Press 'c': Clear all shapes.")
        print("  - Press 's': Save configuration JSON.")
        print("  - Press 'q' or ESC: Quit.\n")

        while True:
            rendered = self.render()
            cv2.imshow(win_name, rendered)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key in (ord('f'), 13):
                self.complete_current_shape()
            elif key == ord('l'):
                self.mode = "lane"
            elif key == ord('z'):
                self.mode = "zone"
            elif key == ord('u'):
                if self.current_points:
                    self.current_points.pop()
            elif key == ord('c'):
                self.current_points.clear()
                self.lanes.clear()
                self.zones.clear()
            elif key == ord('s'):
                self.save_config()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Perspective Polygon Lane & Zone Configurator")
    parser.add_argument("-v", "--video", type=str, default=str(DEFAULT_VIDEO), help="Path to video file")
    args = parser.parse_args()

    drawer = PolygonDrawer(args.video)
    drawer.run()


if __name__ == "__main__":
    main()
