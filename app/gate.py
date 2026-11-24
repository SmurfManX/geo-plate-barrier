"""Virtual barrier: tracks plates across frames and decides once per car.

Per frame the engine returns detections with OCR probabilities, each decoded
against the plate template (weakest character = frame confidence).
Detections are linked into tracks (the same plate moving through the frame).
Every frame of a track votes for the plate it decoded (if conf >= FRAME_CONF).

A decision is made only when, within the last WINDOW frames,
  * the leading plate has at least MIN_AGREE votes, and
  * it holds at least AGREE_SHARE of all votes (a clear majority).
One decision per track (a new one only if the leading plate later changes),
so misreads while the car approaches never reach the log or the barrier.
"""
from __future__ import annotations

import itertools
import logging
import os
from collections import Counter
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import db, plates
from .engine import Read

WINDOW = int(os.environ.get("FUSE_WINDOW", "8"))
MIN_AGREE = int(os.environ.get("MIN_AGREE", "3"))
AGREE_SHARE = float(os.environ.get("AGREE_SHARE", "0.6"))
FRAME_CONF = float(os.environ.get("FRAME_CONF", "0.5"))
SINGLE_CONF = float(os.environ.get("SINGLE_IMAGE_CONF", "0.8"))  # photo upload: one frame only
TRACK_TTL = float(os.environ.get("TRACK_TTL", "2.0"))
OPEN_SECONDS = float(os.environ.get("OPEN_SECONDS", "6"))

_ids = itertools.count(1)
log = logging.getLogger("gate")


@dataclass
class Track:
    box: tuple
    last: float
    id: int = field(default_factory=lambda: next(_ids))
    window: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    frames: int = 0
    decided: str | None = None
    decision: str | None = None
    owner: str = ""
    best: Read | None = None
    fused: plates.Plate | None = None   # leading plate (scored on the best frame)
    agree: int = 0
    votes: int = 0

    def status(self) -> dict:
        state = self.decision or "reading"
        f = self.fused
        return {"track": self.id, "state": state, "frames": self.frames, "agree": self.agree, "votes": self.votes,
                "need": MIN_AGREE, "plate": f.plate if f else None, "display": f.display if f else None,
                "conf": round(f.conf, 3) if f else 0.0, "owner": self.owner}


def _match(box, prev) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    px, py = (prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2
    size = max(box[2] - box[0], prev[2] - prev[0])
    return abs(cx - px) < 0.75 * size and abs(cy - py) < 0.75 * size


class Gate:
    def __init__(self, decoder: plates.Decoder | None = None) -> None:
        self._lock = threading.Lock()
        self.decoder = decoder
        self._tracks: dict[str, list[Track]] = {}
        self.open_until = 0.0
        self.last = None

    def state(self) -> dict:
        now = time.time()
        with self._lock:
            return {"open": now < self.open_until, "closes_in": max(0.0, round(self.open_until - now, 1)),
                    "last": self.last}

    # ---- decision helpers (lock held)
    def _log(self, t: Track, decision: str, source: str, now: float) -> dict:
        b = t.best
        row = db.get_plate(t.decided) if t.decided else None
        t.owner = row["owner"] if row else ""
        event_id = db.add_event(t.decided or "—", decision, raw=b.raw if b else "", owner=t.owner,
                                source=source, conf=t.fused.conf if t.fused else 0.0,
                                crop=b.crop_jpeg if b else None)
        log.info("car #%d %s %s%s | %d of %d frames agree, conf %.2f, source %s, event %d",
                 t.id, plates.display(t.decided) if t.decided else "—", decision.upper(),
                 f" ({t.owner})" if t.owner else "", t.agree, t.votes, t.fused.conf if t.fused else 0.0,
                 source, event_id)
        self.last = {"id": event_id, "ts": now, "plate": t.decided or "",
                     "display": plates.display(t.decided) if t.decided else "",
                     "decision": decision, "owner": t.owner}
        return self.last

    def _update(self, t: Track, r: Read, single: bool) -> bool:
        """Add a frame to the track; return True when a (new) decision is due."""
        t.frames += 1
        t.window.append(r)
        if r.plate and (t.best is None or r.plate.conf >= t.best.ocr_conf):
            t.best = r
        votes = Counter(x.plate.plate for x in t.window if x.plate and x.plate.conf >= FRAME_CONF)
        if not votes:
            t.fused, t.agree, t.votes = None, 0, 0
            return False
        lead, t.agree = votes.most_common(1)[0]
        t.votes = sum(votes.values())
        best = max((x for x in t.window if x.plate and x.plate.plate == lead), key=lambda x: x.plate.conf)
        t.fused = best.plate
        if lead == t.decided:
            return False
        if single:
            return t.fused.conf >= SINGLE_CONF
        return t.agree >= MIN_AGREE and t.agree >= AGREE_SHARE * t.votes

    def observe(self, reads: list[Read], source: str, single: bool = False) -> tuple[list[dict], list[dict]]:
        """Returns (new events, per-read track status aligned with reads)."""
        now = time.time()
        events, statuses = [], []
        with self._lock:
            tracks = [] if single else self._tracks.setdefault(source, [])
            tracks[:] = [t for t in tracks if now - t.last <= TRACK_TTL]
            used = set()
            for r in reads:
                t = next((t for t in tracks if t.id not in used and _match(r.box, t.box)), None)
                if t is None:
                    t = Track(box=r.box, last=now)
                    tracks.append(t)
                used.add(t.id)
                t.box, t.last = r.box, now
                if self._update(t, r, single):
                    t.decided = t.fused.plate
                    row = db.get_plate(t.decided)
                    t.decision = "granted" if row and row["active"] else "denied"
                    events.append(self._log(t, t.decision, source, now))
                if t.decision == "granted":
                    # keep the barrier open while the allowed car is still in view
                    self.open_until = max(self.open_until, now + OPEN_SECONDS)
                statuses.append(t.status())
        return events, statuses

    def manual(self, open_: bool) -> dict:
        now = time.time()
        with self._lock:
            self.open_until = now + OPEN_SECONDS if open_ else 0.0
            decision = "manual_open" if open_ else "manual_close"
            event_id = db.add_event("—", decision, source="operator")
            log.info("operator %s, event %d", "OPEN" if open_ else "CLOSE", event_id)
            self.last = {"id": event_id, "ts": now, "plate": "", "display": "", "decision": decision, "owner": ""}
            return self.last
