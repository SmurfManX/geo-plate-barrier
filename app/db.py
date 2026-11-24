"""SQLite storage: allowed plates and access events."""
from __future__ import annotations

import os
import sqlite3
import threading
import time

DB_PATH = os.environ.get("DB_PATH", "/data/barrier.db")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS plates (
    plate      TEXT PRIMARY KEY,
    owner      TEXT NOT NULL DEFAULT '',
    note       TEXT NOT NULL DEFAULT '',
    active     INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    plate    TEXT NOT NULL,
    raw      TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL,          -- granted | denied | manual_open | manual_close
    owner    TEXT NOT NULL DEFAULT '',
    source   TEXT NOT NULL DEFAULT '',
    conf     REAL NOT NULL DEFAULT 0,
    crop     BLOB
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
    return _conn


def list_plates() -> list[dict]:
    with _lock:
        rows = conn().execute("SELECT * FROM plates ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_plate(plate: str) -> dict | None:
    with _lock:
        row = conn().execute("SELECT * FROM plates WHERE plate = ?", (plate,)).fetchone()
    return dict(row) if row else None


def upsert_plate(plate: str, owner: str = "", note: str = "", active: bool = True) -> None:
    with _lock:
        conn().execute(
            """INSERT INTO plates (plate, owner, note, active, created_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(plate) DO UPDATE SET owner = excluded.owner, note = excluded.note,
                                                active = excluded.active""",
            (plate, owner, note, int(active), time.time()),
        )
        conn().commit()


def seed_plate(plate: str, owner: str = "") -> None:
    with _lock:
        conn().execute(
            "INSERT OR IGNORE INTO plates (plate, owner, created_at) VALUES (?, ?, ?)",
            (plate, owner, time.time()),
        )
        conn().commit()


def delete_plate(plate: str) -> bool:
    with _lock:
        cur = conn().execute("DELETE FROM plates WHERE plate = ?", (plate,))
        conn().commit()
    return cur.rowcount > 0


def add_event(plate: str, decision: str, *, raw: str = "", owner: str = "", source: str = "",
              conf: float = 0.0, crop: bytes | None = None) -> int:
    with _lock:
        cur = conn().execute(
            "INSERT INTO events (ts, plate, raw, decision, owner, source, conf, crop) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), plate, raw, decision, owner, source, conf, crop or None),
        )
        conn().commit()
    return int(cur.lastrowid)


def list_events(limit: int = 50) -> list[dict]:
    with _lock:
        rows = conn().execute(
            "SELECT id, ts, plate, raw, decision, owner, source, conf, crop IS NOT NULL AS has_crop "
            "FROM events ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def event_crop(event_id: int) -> bytes | None:
    with _lock:
        row = conn().execute("SELECT crop FROM events WHERE id = ?", (event_id,)).fetchone()
    return row["crop"] if row and row["crop"] else None
