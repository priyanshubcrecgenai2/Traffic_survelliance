from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PlateOCR:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


class ANPREngine:
    def __init__(self):
        self.char_threshold = 3
        self.char_patterns = {
            "indian": r"^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}$",
        }

    def read_plate(self, plate_crop: np.ndarray) -> PlateOCR:
        if plate_crop is None or plate_crop.size == 0:
            return PlateOCR(text="", confidence=0.0)

        processed = self._preprocess(plate_crop)
        text = self._basic_ocr(processed)
        conf = self._compute_confidence(text)

        return PlateOCR(text=text, confidence=conf)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        denoised = cv2.medianBlur(cleaned, 3)
        return denoised

    def _basic_ocr(self, binary: np.ndarray) -> str:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chars = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 0.1 < aspect < 1.5 and h > binary.shape[0] * 0.3:
                chars.append((x, y, w, h))

        chars.sort(key=lambda c: c[0])
        return f"PLATE_{len(chars)}_CHARS"

    def _compute_confidence(self, text: str) -> float:
        if not text or text.startswith("PLATE_"):
            return 0.3
        if len(text) >= 6:
            return 0.7
        return 0.5

    def read_batch(self, plates: list) -> list[PlateOCR]:
        return [self.read_plate(p.crop if hasattr(p, "crop") else None) for p in plates]
