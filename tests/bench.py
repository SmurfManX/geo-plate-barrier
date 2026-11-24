"""Benchmark: old per-frame logic vs new tracked/fused decoding.

Simulates cars approaching the camera (sequences of webcam-like frames with
blur, noise, glare, perspective, JPEG) and replays them through the real
detector/OCR and both decision algorithms.

  python -m tests.bench --seqs 120 [--real /path/to/crops]

Each sequence: a car with plate P. Half the time P is in the database
(expect: granted P). Otherwise the database holds a *near-miss* of P (one
confusable character changed) - opening for it would be a false grant.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "bench.db"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app import db, engine as eng, gate as gate_mod, plates  # noqa: E402

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
LET, DIG = plates.LETTERS, plates.DIGITS
CONF_L = "OBSIZGTADQ"   # letters that look like digits
CONF_D = "0851267440"   # their digit look-alikes (same index)
SWAP = {**dict(zip(CONF_L, CONF_D)), **{d: l for l, d in zip(CONF_L, CONF_D)}}


# ------------------------------------------------------------------ rendering
def random_plate(rng: random.Random) -> str:
    def letter():
        return rng.choice(CONF_L) if rng.random() < 0.5 else rng.choice(LET)

    def digit():
        return rng.choice(CONF_D) if rng.random() < 0.5 else rng.choice(DIG)
    return letter() + letter() + digit() + digit() + digit() + letter() + letter()


def near_miss(p: str, rng: random.Random) -> str:
    """Same plate with one character replaced by a visually similar one of the same class."""
    similar = {"O": "QDCU", "B": "PRE", "S": "SZ", "I": "TLJ", "Z": "ZS", "G": "COC", "T": "IY", "A": "AR",
               "D": "OB", "Q": "OD", "0": "896", "8": "036", "5": "6S", "1": "7", "2": "7Z", "6": "580", "7": "1",
               "4": "1", "3": "8", "9": "80"}
    for _ in range(50):
        i = rng.randrange(7)
        opts = [c for c in similar.get(p[i], "") if (c in LET) == (p[i] in LET) and c != p[i]]
        if not opts:
            opts = [c for c in (LET if p[i] in LET else DIG) if c != p[i]]
        q = p[:i] + rng.choice(opts) + p[i + 1:]
        if q != p:
            return q
    raise RuntimeError


def plate_image(text: str) -> np.ndarray:
    w, h = 520, 112
    im = Image.new("RGB", (w, h), (238, 238, 236))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((1, 1, w - 2, h - 2), radius=8, outline=(15, 15, 15), width=6)
    d.rectangle((7, 7, 66, h - 8), fill=(20, 70, 175))
    # Georgian flag: white, big red cross, four small crosses
    fx, fy, fw, fh = 15, 15, 44, 30
    d.rectangle((fx, fy, fx + fw, fy + fh), fill=(255, 255, 255))
    d.rectangle((fx + fw // 2 - 3, fy, fx + fw // 2 + 3, fy + fh), fill=(220, 30, 30))
    d.rectangle((fx, fy + fh // 2 - 3, fx + fw, fy + fh // 2 + 3), fill=(220, 30, 30))
    ge = ImageFont.truetype(FONT, 26)
    d.text((17, 62), "GE", font=ge, fill=(255, 255, 255))
    disp = f"{text[:2]}-{text[2:5]}-{text[5:]}"
    font = ImageFont.truetype(FONT, 104)
    tw = d.textlength(disp, font=font)
    txt = Image.new("L", (int(tw) + 10, 120), 0)
    ImageDraw.Draw(txt).text((5, -8), disp, font=font, fill=255)
    txt = txt.resize((w - 96, 92))
    im.paste((15, 15, 15), (80, 10), txt)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def background(rng: np.random.Generator, w=1280, h=720) -> np.ndarray:
    base = rng.integers(40, 160, 3)
    grad = np.linspace(0.6, 1.3, h)[:, None, None] * base[None, None, :]
    img = np.repeat(np.repeat(grad, w, axis=1), 1, axis=2)
    tex = cv2.GaussianBlur(rng.normal(0, 25, (h, w, 3)), (0, 0), 9)
    img = np.clip(img + tex, 0, 255).astype(np.uint8)
    # some lines/blocks so the detector sees clutter
    for _ in range(12):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 300)), y + int(rng.integers(5, 120))), c, -1)
    return img


class Sequence:
    def __init__(self, text: str, seed: int, frames: int = 12):
        self.text, self.frames = text, frames
        r = np.random.default_rng(seed)
        self.r = r
        self.bg = background(r)
        self.car_color = tuple(int(x) for x in r.integers(20, 230, 3))
        self.plate = plate_image(text)
        self.scale0, self.scale1 = r.uniform(0.25, 0.4), r.uniform(0.55, 1.0)
        self.yaw = r.uniform(-0.25, 0.25)       # horizontal perspective
        self.rot = r.uniform(-6, 6)
        self.cx, self.cy = r.uniform(450, 830), r.uniform(380, 520)
        self.dark = r.uniform(0.45, 1.15)
        self.glare = r.random() < 0.35
        self.shadow = r.random() < 0.35

    def frame(self, k: int) -> np.ndarray:
        r = self.r
        t = k / max(1, self.frames - 1)
        s = self.scale0 + (self.scale1 - self.scale0) * t
        img = self.bg.copy()
        ph, pw = self.plate.shape[:2]
        w, h = pw * s, ph * s
        cx, cy = self.cx + r.normal(0, 3), self.cy + 60 * t + r.normal(0, 3)
        cv2.rectangle(img, (int(cx - w * 1.3), int(cy - h * 3.5)), (int(cx + w * 1.3), int(cy + h * 1.6)), self.car_color, -1)
        y = self.yaw * h
        dst = np.float32([[cx - w / 2, cy - h / 2 - y], [cx + w / 2, cy - h / 2 + y],
                          [cx + w / 2, cy + h / 2 - y * 0.6], [cx - w / 2, cy + h / 2 + y * 0.6]])
        m_rot = cv2.getRotationMatrix2D((cx, cy), self.rot, 1.0)
        dst = cv2.transform(dst[None], m_rot)[0].astype(np.float32)
        src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.plate, M, (1280, 720), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (1280, 720))
        img[mask > 0] = warped[mask > 0]
        f = img.astype(np.float32)
        if self.shadow:
            xs = np.linspace(0, 1, 1280)[None, :, None]
            f *= 0.55 + 0.45 * np.clip((xs - 0.3) * 3 + 0.5, 0, 1)
        if self.glare:
            g = np.zeros((720, 1280), np.float32)
            cv2.ellipse(g, (int(cx + r.uniform(-w / 3, w / 3)), int(cy)), (int(w * 0.18), int(h * 0.7)), 20, 0, 360, 1, -1)
            f += cv2.GaussianBlur(g, (0, 0), max(2, w / 25))[..., None] * 150
        f *= self.dark
        # motion blur: stronger when the car is far / moving fast
        mb = int(r.uniform(1, 9) * (1.2 - t))
        if mb >= 2:
            kern = np.zeros((mb, mb), np.float32); kern[mb // 2, :] = 1 / mb
            f = cv2.filter2D(f, -1, kern)
        f = cv2.GaussianBlur(f, (0, 0), r.uniform(0.4, 1.4))
        f += r.normal(0, r.uniform(3, 12), f.shape)
        img = np.clip(f, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(r.uniform(55, 90))])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ old algorithm (v1)
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}


def old_parse(text: str, templates=("LL-DDD-LL", "LLL-DDD")):
    s = re.sub(r"[^A-Z0-9]", "", text.upper())
    best, best_score = None, None
    for tpl in templates:
        pattern = tpl.replace("-", "")
        n = len(pattern)
        for start in range(0, len(s) - n + 1):
            out, fixes = [], 0
            for ch, kind in zip(s[start:start + n], pattern):
                if kind == "L":
                    if ch.isalpha(): out.append(ch)
                    elif ch in TO_LETTER: out.append(TO_LETTER[ch]); fixes += 1
                    else: out = None; break
                else:
                    if ch.isdigit(): out.append(ch)
                    elif ch in TO_DIGIT: out.append(TO_DIGIT[ch]); fixes += 1
                    else: out = None; break
            if out is None or fixes > 2:
                continue
            rest = s[:start].removesuffix("GE") + s[start + n:]
            score = fixes + len(rest)
            if best is None or score < best_score:
                best, best_score = "".join(out), score
    return best


class OldGate:
    """v1: plate seen in 2 frames within 3 s -> decide; same plate again only after 20 s."""

    def __init__(self, allowed: set[str]):
        self.allowed, self.hits, self.last_event, self.decisions = allowed, defaultdict(deque), {}, []

    def observe(self, texts_confs: list[tuple[str, float]], now: float, k: int = 0):
        for text, conf in texts_confs:
            p = old_parse(text) if conf >= 0.5 else None
            if not p:
                continue
            h = self.hits[p]; h.append(now)
            while h and now - h[0] > 3: h.popleft()
            if len(h) >= 2 and now - self.last_event.get(p, -1e9) >= 20:
                self.last_event[p] = now
                self.decisions.append((p, "granted" if p in self.allowed else "denied", k))


# ------------------------------------------------------------------ harness
class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


def run(seqs: int, seed: int, pad_sets: dict[str, list[float]], frame_gap: float = 0.15):
    e = eng.Engine()
    rng = random.Random(seed)
    clock = Clock()
    gate_mod.time.time = clock  # simulated time for the new gate
    all_pads = sorted({p for ps in pad_sets.values() for p in ps})
    results = {name: Counter() for name in ["old", *pad_sets]}
    latency = {name: [] for name in ["old", *pad_sets]}
    frame_ok = {name: Counter() for name in ["old", *pad_sets]}
    det_frames = total_frames = 0
    examples = defaultdict(list)
    t_start = time.time()
    for si in range(seqs):
        truth = random_plate(rng)
        positive = si % 2 == 0
        in_db = truth if positive else near_miss(truth, rng)
        db.conn().execute("DELETE FROM plates"); db.conn().execute("DELETE FROM events"); db.conn().commit()
        db.upsert_plate(in_db, "bench")
        seq = Sequence(truth, seed * 1000 + si)
        old = OldGate({in_db})
        gates = {name: gate_mod.Gate(e.decoder) for name in pad_sets}
        new_dec = {name: [] for name in pad_sets}
        for k in range(seq.frames):
            clock.t += frame_gap
            frame = seq.frame(k)
            total_frames += 1
            dets = [d for d in e._alpr.detector.predict(frame) if d.confidence >= eng.MIN_DET_CONF]
            det_frames += bool(dets)
            per_det = []
            h, w = frame.shape[:2]
            for d in dets:
                bb = d.bounding_box
                box = (max(0, int(bb.x1)), max(0, int(bb.y1)), min(w, int(bb.x2)), min(h, int(bb.y2)))
                if box[2] - box[0] < 8 or box[3] - box[1] < 4:
                    continue
                probs, _ = e.ocr(e.crops(frame, box, all_pads))
                per_det.append((box, d, dict(zip(all_pads, probs))))
            # old: tight crop argmax text
            texts = []
            for box, d, pp in per_det:
                p0 = pp[0.0]
                raw = "".join(e.alphabet[i] for i in p0.argmax(1)).replace(e.pad_char, "")
                texts.append((raw, float(p0.max(1)[: len(raw)].mean()) if raw else 0.0))
                frame_ok["old"]["n"] += 1
                frame_ok["old"]["ok"] += old_parse(raw) == truth
            old.observe(texts, clock.t, k)
            # new variants
            for name, pads in pad_sets.items():
                reads = []
                for box, d, pp in per_det:
                    logp = np.mean([np.log(np.clip(pp[p], 1e-9, 1)) for p in pads], axis=0)
                    p = np.exp(logp - logp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                    dec = e.decoder.decode(p)
                    reads.append(eng.Read(raw="", probs=p, plate=dec, det_conf=d.confidence, box=box,
                                          region=None, region_prob=0.0, crop_jpeg=b""))
                    frame_ok[name]["n"] += 1
                    frame_ok[name]["ok"] += bool(dec and dec.plate == truth and dec.conf >= gate_mod.FRAME_CONF)
                evs, _ = gates[name].observe(reads, "bench")
                new_dec[name] += [(ev["plate"], ev["decision"], k) for ev in evs]
        # score
        def score(name, decisions):
            want = "granted" if positive else "denied"
            first_ok = bool(decisions) and decisions[0][0] == truth and decisions[0][1] == want
            false_grant = any(dd == "granted" and p != truth for p, dd, *_ in decisions) or \
                (not positive and any(dd == "granted" for p, dd, *_ in decisions))
            wrong_events = sum(1 for p, *_ in decisions if p != truth)
            c = results[name]
            c["seqs"] += 1
            c["correct"] += first_ok and not false_grant and wrong_events == 0
            c["false_grant"] += false_grant
            c["missed"] += not decisions
            c["wrong_plate_events"] += wrong_events
            if not (first_ok and not false_grant and wrong_events == 0) and len(examples[name]) < 6:
                examples[name].append((truth, in_db, decisions[:4]))
            if first_ok:
                latency[name].append(decisions[0][2])
        score("old", old.decisions)
        for name in pad_sets:
            score(name, new_dec[name])
    print(f"\n{seqs} sequences x 12 frames, detector found a plate in {det_frames}/{total_frames} frames, "
          f"{time.time() - t_start:.0f}s")
    print(f"{'algorithm':<22}{'correct':>9}{'false grant':>13}{'no decision':>13}{'wrong-plate events':>20}"
          f"{'single-frame acc':>18}{'frames to decide':>18}")
    for name, c in results.items():
        fo = frame_ok[name]
        lat = np.median(latency[name]) + 1 if latency[name] else float("nan")
        print(f"{name:<22}{c['correct']:>5}/{c['seqs']:<3}{c['false_grant']:>13}{c['missed']:>13}"
              f"{c['wrong_plate_events']:>20}{fo['ok'] / max(1, fo['n']):>17.1%}{lat:>18.1f}")
    for name, ex in examples.items():
        for truth, in_db, decs in ex:
            print(f"  [{name}] truth={truth} db={in_db} decisions={decs}")


def real_crops(path: str, truth: str):
    e = eng.Engine()
    files = sorted(glob.glob(os.path.join(path, "*.jpg")))
    print(f"\nreal webcam crops ({len(files)}), truth {truth}:")
    for f in files:
        img = cv2.imread(f)
        probs, _ = e.ocr([img])
        p = probs[0]
        raw = "".join(e.alphabet[i] for i in p.argmax(1)).replace(e.pad_char, "")
        dec = e.decoder.decode(p)
        print(f"  {os.path.basename(f):<14} raw={raw:<10} old={str(old_parse(raw)):<9} "
              f"new={dec.plate if dec else None} conf={dec.conf if dec else 0:.2f} "
              f"{'OK' if dec and dec.plate == truth and dec.conf >= gate_mod.FRAME_CONF else ('rejected' if dec and dec.conf < gate_mod.FRAME_CONF else 'WRONG')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--real", default="")
    ap.add_argument("--real-truth", default="NO717AH")
    ap.add_argument("--sample", default="", help="write a few frames of one sequence to this dir and exit")
    a = ap.parse_args()
    if a.sample:
        os.makedirs(a.sample, exist_ok=True)
        s = Sequence("NO717AH", 7)
        for k in (0, 5, 11):
            cv2.imwrite(os.path.join(a.sample, f"seq_{k:02d}.jpg"), s.frame(k))
        sys.exit()
    if a.real:
        real_crops(a.real, a.real_truth)
    run(a.seqs, a.seed, {"new: tight crop": [0.0], "new: tight+pad6%": [0.0, 0.06],
                         "new: 3 crops": [0.0, 0.06, 0.12]})
