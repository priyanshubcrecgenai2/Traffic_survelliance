#!/usr/bin/env python3
"""
Interactive Polygon Lane and Region Drawer Tool for Traffic Intelligence System.
Allows drawing angled/perspective 4-point polygon lanes and monitoring regions.

Usage:
    python draw_lanes.py -v traffic_2.mp4
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path

from config import DEFAULT_VIDEO, OUTPUT_DIR


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
        self.lanes: list[dict] = []  # {"name": str, "direction": str, "polygon": [(x, y), ...]}
        self.zones: list[dict] = []  # {"name": str, "polygon": [(x, y), ...]}
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
                    "polygon": list(self.current_points),
                })
                print(f"[LANE ADDED] lane_{lane_num} with {len(self.current_points)} vertices ({direction})")
            elif self.mode == "zone":
                zone_num = len(self.zones) + 1
                self.zones.append({
                    "name": f"zone_{zone_num}",
                    "polygon": list(self.current_points),
                })
                print(f"[ZONE ADDED] zone_{zone_num} with {len(self.current_points)} vertices")
            self.current_points.clear()
        else:
            print("[WARN] Need at least 3 points to complete a polygon region")

    def render(self) -> np.ndarray:
        vis = self.orig_frame.copy()
        overlay = vis.copy()

        # Render finished lanes
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

        # Render finished zones
        for z in self.zones:
            pts = np.array(z["polygon"], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
            cx = int(np.mean([p[0] for p in z["polygon"]]))
            cy = int(np.mean([p[1] for p in z["polygon"]]))
            cv2.putText(vis, z["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Blend semi-transparent polygons
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # Render currently drawing points & lines
        if self.current_points:
            active_c = (0, 255, 0) if self.mode == "lane" else (0, 255, 255)
            for pt in self.current_points:
                cv2.circle(vis, pt, 5, active_c, -1)
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(vis, [pts], False, active_c, 2)

        # Top HUD instructions
        hud = np.zeros((80, self.w, 3), dtype=np.uint8)
        mode_str = f"MODE: {'POLYGON LANE' if self.mode == 'lane' else 'MONITOR ZONE'}"
        pts_str = f"Current Vertices: {len(self.current_points)}"
        cv2.putText(hud, f"{mode_str} | {pts_str} | [L]ane | [Z]one | [F]inish Shape | [U]ndo | [C]lear | [S]ave | [Q]uit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(hud, "Click points around each lane/region corners -> Press 'F' or ENTER to finish shape",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return np.vstack([hud, vis])

    def save_config(self) -> str:
        config_data = {
            "frame_width": self.w,
            "frame_height": self.h,
            "lanes": self.lanes,
            "zones": self.zones,
        }

        out_file = Path(OUTPUT_DIR) / "lanes_config.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(config_data, f, indent=2)

        print("\n" + "=" * 65)
        print(f"[SAVED] Configuration saved to {out_file}")
        print("=" * 65)
        print("\n--- Python Snippet for config.py ---")
        print("from config import LaneConfig\n")
        print("CUSTOM_LANES = [")
        for lane in self.lanes:
            poly_str = str(lane["polygon"])
            print(f'    LaneConfig(name="{lane["name"]}", direction="{lane["direction"]}", polygon={poly_str}),')
        print("]\n")
        if self.zones:
            print("MONITOR_ZONES = {")
            for z in self.zones:
                print(f'    "{z["name"]}": {z["polygon"]},')
            print("}")
        print("=" * 65 + "\n")
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
        print("  - Press 'u': Undo last vertex or shape.")
        print("  - Press 'c': Clear all shapes.")
        print("  - Press 's': Save configuration JSON and print python snippet.")
        print("  - Press 'q' or ESC: Quit.\n")

        while True:
            rendered = self.render()
            cv2.imshow(win_name, rendered)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), 27):  # q or ESC
                break
            elif key in (ord('f'), 13):  # f or ENTER
                self.complete_current_shape()
            elif key == ord('l'):
                self.mode = "lane"
                print("[MODE] Polygon Lane mode")
            elif key == ord('z'):
                self.mode = "zone"
                print("[MODE] Monitor Zone mode")
            elif key == ord('u'):
                if self.current_points:
                    self.current_points.pop()
                    print("[UNDO] Removed last vertex point")
                elif self.mode == "lane" and self.lanes:
                    popped = self.lanes.pop()
                    print(f"[UNDO] Removed {popped['name']}")
                elif self.mode == "zone" and self.zones:
                    popped = self.zones.pop()
                    print(f"[UNDO] Removed {popped['name']}")
            elif key == ord('c'):
                self.current_points.clear()
                self.lanes.clear()
                self.zones.clear()
                print("[CLEAR] Cleared all polygons")
            elif key == ord('s'):
                self.save_config()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive Perspective Polygon Lane & Zone Configurator")
    parser.add_argument("-v", "--video", type=str, default=str(DEFAULT_VIDEO), help="Path to video file")
    args = parser.parse_args()

    drawer = PolygonDrawer(args.video)
    drawer.run()


if __name__ == "__main__":
    main()
