"""Plate detection + OCR.

Detector: YOLOv9 from fast-alpr. OCR: fast-plate-ocr, but we keep the full
probability matrix (slots x alphabet) instead of only the argmax string, so
the plate can be decoded against the Georgian template and fused over frames.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from fast_alpr import ALPR
from fast_plate_ocr.inference import plate_recognizer as fpr

from . import plates

DETECTOR = os.environ.get("ALPR_DETECTOR", "yolo-v9-t-512-license-plate-end2end")
OCR = os.environ.get("ALPR_OCR", "cct-s-v2-global-model")
MIN_DET_CONF = float(os.environ.get("MIN_DET_CONF", "0.35"))
# Crop variants OCR'd per detection and averaged (test-time augmentation).
# Each value is extra padding as a fraction of the box size; "0" = tight box.
TTA_PADS = [float(x) for x in os.environ.get("OCR_TTA_PADS", "0,0.06").split(",") if x.strip()]


@dataclass
class Read:
    raw: str                     # argmax text, for debugging only
    probs: np.ndarray            # (slots, alphabet) probabilities
    plate: plates.Plate | None   # single-frame decode against the templates
    det_conf: float
    box: tuple[int, int, int, int]
    region: str | None
    region_prob: float           # P(region == Georgia)
    crop_jpeg: bytes

    @property
    def ocr_conf(self) -> float:
        return self.plate.conf if self.plate else 0.0


class Engine:
    def __init__(self) -> None:
        self._alpr = ALPR(detector_model=DETECTOR, ocr_model=OCR,
                          detector_conf_thresh=MIN_DET_CONF, ocr_device="cpu")
        self._rec = self._alpr.ocr.ocr_model
        cfg = self._rec.config
        self.alphabet: str = cfg.alphabet
        self.pad_char: str = cfg.pad_char
        self._rgb = cfg.image_color_mode == "rgb"
        self._outputs = self._rec._fetch_outputs if self._rec.has_region_head else [self._rec.plate_output_name]
        regions = list(cfg.plate_regions or []) if self._rec.has_region_head else []
        self._ge = regions.index("Georgia") if "Georgia" in regions else None
        self._regions = regions
        self.decoder = plates.Decoder(self.alphabet, self.pad_char)
        self._lock = threading.Lock()

    # -- OCR on a batch of BGR crops -> (N, slots, alphabet), (N, regions) or None
    def ocr(self, crops: list[np.ndarray]):
        imgs = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB if self._rgb else cv2.COLOR_BGR2GRAY) for c in crops]
        x = fpr.preprocess_image(fpr._load_image_from_source(imgs, self._rec.config))
        out = dict(zip(self._outputs, self._rec.model.run(self._outputs, {"input": x})))
        probs = out[self._rec.plate_output_name].reshape(len(crops), self._rec.config.max_plate_slots, -1)
        region = out.get(self._rec.region_output_name) if self._rec.has_region_head else None
        return probs.astype(np.float64), region

    @staticmethod
    def crops(frame: np.ndarray, box, pads) -> list[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        out = []
        for p in pads:
            dx, dy = int(bw * p), int(bh * p)
            out.append(frame[max(0, y1 - dy):min(h, y2 + dy), max(0, x1 - dx):min(w, x2 + dx)])
        return out

    def read_crop_variants(self, variants: list[np.ndarray]):
        probs, region = self.ocr(variants)
        # Geometric mean over variants (product of experts), renormalised per slot.
        logp = np.log(np.clip(probs, 1e-9, 1.0)).mean(axis=0)
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        reg_name, ge_prob = None, 0.0
        if region is not None:
            r = region.mean(axis=0)
            reg_name = self._regions[int(r.argmax())]
            ge_prob = float(r[self._ge]) if self._ge is not None else 0.0
        return p, reg_name, ge_prob

    def read(self, frame: np.ndarray) -> list[Read]:
        with self._lock:
            dets = self._alpr.detector.predict(frame)
            out: list[Read] = []
            h, w = frame.shape[:2]
            for det in dets:
                if det.confidence < MIN_DET_CONF:
                    continue
                bb = det.bounding_box
                box = (max(0, int(bb.x1)), max(0, int(bb.y1)), min(w, int(bb.x2)), min(h, int(bb.y2)))
                if box[2] - box[0] < 8 or box[3] - box[1] < 4:
                    continue
                variants = self.crops(frame, box, TTA_PADS)
                p, reg_name, ge_prob = self.read_crop_variants(variants)
                raw = "".join(self.alphabet[i] for i in p.argmax(axis=1)).rstrip(self.pad_char).replace(self.pad_char, "")
                ok, buf = cv2.imencode(".jpg", variants[0], [cv2.IMWRITE_JPEG_QUALITY, 85])
                out.append(Read(raw=raw, probs=p, plate=self.decoder.decode(p), det_conf=float(det.confidence),
                                box=box, region=reg_name, region_prob=ge_prob,
                                crop_jpeg=buf.tobytes() if ok else b""))
        return out
