#!/usr/bin/env python3
"""
ANPR & Plate Detector Verification Script
Tests license-plate-finetune-v1l.pt and ANPREngine on traffic_2.mp4
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from config import MODELS_DIR, DEVICE

def test_anpr():
    video_path = "traffic_2.mp4"
    if not Path(video_path).is_file():
        video_path = "traffic_1.mp4"
    
    plate_model_path = str(MODELS_DIR / "license-plate-finetune-v1l.pt")
    print(f"[TEST] Loading plate detector from: {plate_model_path}")
    print(f"[TEST] Using device: {DEVICE}")
    print(f"[TEST] Opening video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return
    
    model = YOLO(plate_model_path)
    
    output_dir = Path("output/anpr_test_crops")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_frames = 0
    detected_plates_count = 0
    crops_saved = 0
    plate_sizes = []
    confidences = []
    
    start_time = time.time()
    
    frame_idx = 0
    while frame_idx < 300: # test first 300 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        total_frames += 1
        
        # Test plate detection
        results = model(frame, conf=0.25, device=DEVICE, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    w, h = x2 - x1, y2 - y1
                    detected_plates_count += 1
                    plate_sizes.append((w, h))
                    confidences.append(conf)
                    
                    if crops_saved < 20 and w > 5 and h > 5:
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        if crop.size > 0:
                            crop_file = output_dir / f"plate_f{frame_idx}_{crops_saved}_w{w}h{h}_c{conf:.2f}.jpg"
                            cv2.imwrite(str(crop_file), crop)
                            crops_saved += 1
                            
    cap.release()
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("ANPR / LICENSE PLATE EVALUATION REPORT")
    print("="*50)
    print(f"Total Frames Tested: {total_frames}")
    print(f"Total Plate Detections: {detected_plates_count}")
    print(f"Plates per Frame: {detected_plates_count / max(1, total_frames):.3f}")
    print(f"Avg Detection Time per frame: {(elapsed/max(1, total_frames))*1000:.1f} ms")
    if plate_sizes:
        avg_w = np.mean([s[0] for s in plate_sizes])
        avg_h = np.mean([s[1] for s in plate_sizes])
        avg_conf = np.mean(confidences)
        print(f"Avg Plate Resolution: {avg_w:.1f} x {avg_h:.1f} pixels")
        print(f"Avg Plate Confidence: {avg_conf:.2f}")
        print(f"Saved {crops_saved} sample crops to: {output_dir}")
    else:
        print("No plates detected in the test sample.")
    print("="*50)

if __name__ == "__main__":
    test_anpr()
