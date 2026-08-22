from pathlib import Path
from dataclasses import dataclass, field
import json
import torch


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

VEHICLE_MODEL = MODELS_DIR / "UVH-26-MV-YOLOv11-S.pt"
PLATE_MODEL = MODELS_DIR / "license-plate-finetune-v1l.pt"
DEFAULT_VIDEO = BASE_DIR / "traffic_1.mp4"

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

CLASSES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "auto_rickshaw",
    5: "bicycle",
    6: "pedestrian",
    7: "van",
    8: "minivan",
    9: "suv",
}

VEHICLE_CLASSES = CLASSES
PEDESTRIAN_CLASS_IDS = [6]
CYCLIST_CLASS_IDS = [5]

DIRECTIONS = {"T2B": "top_to_bottom", "B2T": "bottom_to_top"}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TRACK_THRESH = 0.25
MATCH_THRESH = 0.8


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
    lanes: list = field(default_factory=list)
    zones: dict = field(default_factory=dict)
    output_dir: str = str(OUTPUT_DIR)
    save_annotated: bool = True
    save_crops: bool = True
    show_live: bool = True
    device: str = DEVICE
    regions_path: str = str(OUTPUT_DIR / "lanes_config.json")
    filter_by_regions: bool = True
    dashboard_port: int = 5000


def load_regions_from_json(json_path: str | Path) -> tuple[list[LaneConfig], dict]:
    path = Path(json_path)
    if not path.is_file():
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
        for z in data.get("zones", []):
            z_name = z.get("name", "zone")
            if "polygon" in z:
                zones[z_name] = [(int(p[0]), int(p[1])) for p in z["polygon"]]
            elif "bbox" in z:
                zones[z_name] = tuple(map(int, z["bbox"]))
        return lanes, zones
    except Exception as e:
        print(f"[WARN] Failed to load regions from {json_path}: {e}")
        return [], {}


def get_default_lanes(frame_width: int) -> list[LaneConfig]:
    third = frame_width / 3
    return [
        LaneConfig("lane_1", 0, third, "B2T"),
        LaneConfig("lane_2", third, 2 * third, "T2B"),
        LaneConfig("lane_3", 2 * third, frame_width, "B2T"),
    ]
