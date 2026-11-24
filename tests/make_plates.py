"""Generate synthetic Georgian plates on a noisy background (for offline testing).

python -m tests.make_plates OUT_DIR KA-505-BA GU-127-GU ...
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def plate_img(text: str) -> np.ndarray:
    h, w = 112, 520
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (20, 20, 20), 6)
    # blue strip with flag and GE
    cv2.rectangle(img, (6, 6), (62, h - 7), (170, 70, 20), -1)
    cv2.rectangle(img, (14, 16), (54, 44), (255, 255, 255), -1)
    cv2.rectangle(img, (31, 16), (37, 44), (40, 40, 220), -1)
    cv2.rectangle(img, (14, 27), (54, 33), (40, 40, 220), -1)
    cv2.putText(img, "GE", (14, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # characters, condensed like the real DIN-style font
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 2.6, 7
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    canvas = np.full((th + 30, tw + 20, 3), 235, np.uint8)
    cv2.putText(canvas, text, (10, th + 12), font, scale, (15, 15, 15), thick, cv2.LINE_AA)
    canvas = cv2.resize(canvas, (w - 90, h - 24))
    img[12:12 + canvas.shape[0], 76:76 + canvas.shape[1]] = canvas
    return img


def scene(text: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = rng.integers(60, 120, (720, 1280, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    # a dark "car body" behind the plate
    cv2.rectangle(bg, (300, 250), (980, 650), (50, 45, 40), -1)
    plate = plate_img(text)
    ph, pw = plate.shape[:2]
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    x, y = 380, 440
    dst = np.float32([[x, y], [x + pw, y + 12], [x + pw - 4, y + ph + 14], [x + 2, y + ph]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, m, (1280, 720))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (1280, 720))
    bg[mask > 0] = warped[mask > 0]
    noise = rng.normal(0, 6, bg.shape)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(sys.argv[2:]):
        cv2.imwrite(str(out / f"{t}.jpg"), scene(t, i))
        print(out / f"{t}.jpg")
