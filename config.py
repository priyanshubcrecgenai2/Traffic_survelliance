from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
import json
import torch


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
REGIONS_CONFIG_PATH = CONFIG_DIR / "regions.json"

VEHICLE_MODEL = MODELS_DIR / "UVH-26-MV-YOLOv11-S.pt"
PLATE_MODEL = MODELS_DIR / "license-plate-finetune-v1l.pt"
DEFAULT_VIDEO = BASE_DIR / "traffic_2.mp4"

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


@dataclass
class LaneConfig:
    name: str
    x_start: float = 0.0
    x_end: float = 0.0
    direction: str = "T2B"
    polygon: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    video_path: str = str(DEFAULT_VIDEO)
    vehicle_model: str = str(VEHICLE_MODEL)
    plate_model: str = str(PLATE_MODEL)
    confidence: float = CONFIDENCE_THRESHOLD
    iou: float = IOU_THRESHOLD
    lanes: list[LaneConfig] = field(default_factory=list)
    zones: dict = field(default_factory=dict)
    output_dir: str = str(OUTPUT_DIR)
    save_annotated: bool = True
    show_live: bool = False
    device: str = DEVICE
    regions_path: str = str(REGIONS_CONFIG_PATH)
    filter_by_regions: bool = True
    dashboard_port: int = 5000


def load_regions_from_json(json_path: str | Path) -> tuple[list[LaneConfig], dict]:
    path = Path(json_path)
    if not path.is_file():
        # Fallback to output/lanes_config.json if config/regions.json is not found
        fallback = OUTPUT_DIR / "lanes_config.json"
        if fallback.is_file():
            path = fallback
        else:
            return [], {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        lanes = []
        for l in data.get("lanes", []):
            lanes.append(LaneConfig(
                name=l.get("name", "lane"),
                x_start=float(l.get("x_start", 0.0)),
                x_end=float(l.get("x_end", 0.0)),
                direction=l.get("direction", "T2B"),
                polygon=[(int(p[0]), int(p[1])) for p in l.get("polygon", [])],
            ))
        zones = {}
        raw_zones = data.get("zones", {})
        if isinstance(raw_zones, dict):
            zones = {k: [(int(p[0]), int(p[1])) for p in v] for k, v in raw_zones.items()}
        elif isinstance(raw_zones, list):
            for z in raw_zones:
                z_name = z.get("name", "zone")
                if "polygon" in z:
                    zones[z_name] = [(int(p[0]), int(p[1])) for p in z["polygon"]]
        return lanes, zones
    except Exception as e:
        print(f"[CONFIG WARN] Could not load regions from {json_path}: {e}")
        return [], {}


def get_default_lanes(frame_width: int) -> list[LaneConfig]:
    mid = frame_width // 2
    return [
        LaneConfig(name="lane_1", x_start=0, x_end=mid, direction="B2T"),
        LaneConfig(name="lane_2", x_start=mid, x_end=frame_width, direction="T2B"),
    ]
