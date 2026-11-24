"""HTTP API + web UI for the plate-recognition barrier."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, plates
from .camera import ServerCamera
from .engine import Engine, Read
from .gate import Gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC = Path(__file__).parent / "static"
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1280"))

engine: Engine
gate = Gate()
camera: ServerCamera


def seed() -> None:
    # SEED_PLATES="BU-100-LI:Owner,AA-001-AA" — added once, never overwrites edits made in the UI.
    for item in os.environ.get("SEED_PLATES", "").split(","):
        if not item.strip():
            continue
        plate_txt, _, owner = item.partition(":")
        p = plates.normalize_user_input(plate_txt)
        if p:
            db.seed_plate(p, owner.strip())
        else:
            log.warning("SEED_PLATES: %r is not a valid plate", plate_txt)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, camera
    db.conn()
    seed()
    t0 = time.time()
    engine = Engine()
    gate.decoder = engine.decoder
    log.info("ALPR models loaded in %.1fs", time.time() - t0)
    camera = ServerCamera(engine, gate)
    camera.start()
    yield
    camera.stop()


app = FastAPI(title="Plate barrier", lifespan=lifespan)


@app.middleware("http")
async def no_stale_ui(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"  # always revalidate (ETag) so UI updates show up
    return resp


def serialize_reads(reads: list[Read], statuses: list[dict]) -> list[dict]:
    out = []
    for r, st in zip(reads, statuses):
        out.append({
            "raw": r.raw,
            "frame_plate": r.plate.display if r.plate else None,
            "frame_conf": round(r.ocr_conf, 3),
            "det_conf": round(r.det_conf, 3),
            "region": r.region,
            "box": r.box,
            **st,  # track: state (reading|granted|denied), fused plate/display/conf, frames, agree
        })
    return out


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        k = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return img


@app.post("/api/recognize")
async def recognize(request: Request, source: str = "browser-camera"):
    """Body: raw JPEG/PNG bytes. source=upload decides on a single frame."""
    img = decode(await request.body())
    t0 = time.time()
    reads = await asyncio.to_thread(engine.read, img)
    ms = int((time.time() - t0) * 1000)
    events, statuses = gate.observe(reads, source=source, single=source == "upload")
    h, w = img.shape[:2]
    return {"w": w, "h": h, "ms": ms, "reads": serialize_reads(reads, statuses), "events": events,
            "gate": gate.state()}


@app.get("/api/state")
def state():
    return {
        "gate": gate.state(),
        "server_camera": {"enabled": camera.enabled, "status": camera.status, "last": camera.reads or None},
    }


@app.post("/api/gate/{action}")
def gate_action(action: str):
    if action not in ("open", "close"):
        raise HTTPException(404)
    gate.manual(action == "open")
    return gate.state()


class PlateIn(BaseModel):
    plate: str
    owner: str = ""
    note: str = ""
    active: bool = True


@app.get("/api/plates")
def get_plates():
    return [{**p, "display": plates.display(p["plate"])} for p in db.list_plates()]


@app.post("/api/plates")
def add_plate(body: PlateIn):
    p = plates.normalize_user_input(body.plate)
    if not p:
        raise HTTPException(422, f"Invalid plate format: {body.plate!r}. Expected {', '.join(plates.templates())}")
    db.upsert_plate(p, body.owner.strip(), body.note.strip(), body.active)
    return {"plate": p, "display": plates.display(p)}


@app.delete("/api/plates/{plate}")
def remove_plate(plate: str):
    if not db.delete_plate(plates.clean(plate)):
        raise HTTPException(404)
    return {"ok": True}


@app.get("/api/events")
def get_events(limit: int = 50):
    return [{**e, "display": plates.display(e["plate"])} for e in db.list_events(min(limit, 500))]


@app.get("/api/events/{event_id}/crop.jpg")
def get_crop(event_id: int):
    data = db.event_crop(event_id)
    if not data:
        raise HTTPException(404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/camera/stream")
async def camera_stream():
    if not camera.enabled:
        raise HTTPException(404, "CAMERA_SOURCE is not set")

    async def gen():
        while True:
            if camera.jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + camera.jpeg + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
