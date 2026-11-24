"""Server-side camera (for the Linux box): USB device or RTSP/HTTP stream via OpenCV.

Enabled when CAMERA_SOURCE is set: "0" or "/dev/video0" for USB,
"rtsp://user:pass@10.0.0.5:554/stream1" for an IP camera.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("camera")

SOURCE = os.environ.get("CAMERA_SOURCE", "").strip()
PROCESS_INTERVAL = float(os.environ.get("PROCESS_INTERVAL", "0.3"))


class ServerCamera:
    def __init__(self, engine, gate) -> None:
        self.engine, self.gate = engine, gate
        self.frame: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.reads: dict | None = None
        self.status = "off"
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(SOURCE)

    def start(self) -> None:
        if not self.enabled:
            return
        threading.Thread(target=self._capture_loop, daemon=True, name="capture").start()
        threading.Thread(target=self._process_loop, daemon=True, name="process").start()

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> cv2.VideoCapture:
        src: int | str = int(SOURCE) if SOURCE.isdigit() else SOURCE
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.status = f"cannot open {SOURCE}, retrying"
                log.warning(self.status)
                time.sleep(3)
                continue
            self.status = "ok"
            log.info("camera %s opened", SOURCE)
            last_jpeg = 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.status = "stream lost, reconnecting"
                    log.warning(self.status)
                    break
                self.frame = frame
                now = time.time()
                if now - last_jpeg > 0.1:  # preview ~10 fps
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        self.jpeg = buf.tobytes()
                    last_jpeg = now
            cap.release()
            time.sleep(1)

    def _process_loop(self) -> None:
        from .main import serialize_reads  # avoid import cycle at module load

        while not self._stop.is_set():
            frame = self.frame
            if frame is None:
                time.sleep(0.2)
                continue
            t0 = time.time()
            try:
                reads = self.engine.read(frame)
                _, statuses = self.gate.observe(reads, source="server-camera")
                h, w = frame.shape[:2]
                self.reads = {"w": w, "h": h, "ts": t0, "reads": serialize_reads(reads, statuses)}
            except Exception:
                log.exception("recognition failed")
            time.sleep(max(0.0, PROCESS_INTERVAL - (time.time() - t0)))
