"""Tune the voting parameters: OCR every frame once, then replay the decision logic.

python -m tests.sweep --seqs 200 [--cache samples/sweep.pkl]
"""
import argparse
import itertools
import os
import pickle
import random
from collections import Counter

import numpy as np

from tests.bench import Clock, Sequence, near_miss, random_plate  # noqa: F401  (sets DB_PATH)
from app import db, engine as eng, gate as gate_mod


def collect(seqs: int, seed: int):
    e = eng.Engine()
    rng = random.Random(seed)
    out = []
    for si in range(seqs):
        truth = random_plate(rng)
        positive = si % 2 == 0
        in_db = truth if positive else near_miss(truth, rng)
        seq = Sequence(truth, seed * 1000 + si)
        frames = []
        for k in range(seq.frames):
            frame = seq.frame(k)
            h, w = frame.shape[:2]
            dets = []
            for d in e._alpr.detector.predict(frame):
                if d.confidence < eng.MIN_DET_CONF:
                    continue
                bb = d.bounding_box
                box = (max(0, int(bb.x1)), max(0, int(bb.y1)), min(w, int(bb.x2)), min(h, int(bb.y2)))
                if box[2] - box[0] < 8 or box[3] - box[1] < 4:
                    continue
                p, _, _ = e.read_crop_variants(e.crops(frame, box, [0.0]))
                dets.append((box, float(d.confidence), p.astype(np.float32)))
            frames.append(dets)
        out.append({"truth": truth, "db": in_db, "positive": positive, "frames": frames})
        if si % 20 == 0:
            print(f"  collected {si}/{seqs}", flush=True)
    return out, e.decoder


def evaluate(cache, decoder, params):
    for k, v in params.items():
        setattr(gate_mod, k, v)
    clock = Clock()
    gate_mod.time.time = clock
    c = Counter()
    lat = []
    for s in cache:
        db.conn().execute("DELETE FROM plates"); db.conn().commit()
        db.upsert_plate(s["db"], "bench")
        g = gate_mod.Gate(decoder)
        decisions = []
        for k, dets in enumerate(s["frames"]):
            clock.t += 0.15
            reads = [eng.Read(raw="", probs=p.astype(np.float64), plate=decoder.decode(p.astype(np.float64)),
                              det_conf=dc, box=box, region=None, region_prob=0.0, crop_jpeg=b"")
                     for box, dc, p in dets]
            evs, _ = g.observe(reads, "bench")
            decisions += [(ev["plate"], ev["decision"], k) for ev in evs]
        want = "granted" if s["positive"] else "denied"
        first_ok = bool(decisions) and decisions[0][0] == s["truth"] and decisions[0][1] == want
        wrong = sum(1 for p, *_ in decisions if p != s["truth"])
        fg = any(d == "granted" for p, d, _ in decisions if p != s["truth"] or not s["positive"])
        c["n"] += 1
        c["correct"] += first_ok and wrong == 0 and not fg
        c["false_grant"] += fg
        c["no_decision"] += not decisions
        c["wrong_events"] += wrong
        if first_ok:
            lat.append(decisions[0][2] + 1)
    c["latency"] = float(np.median(lat)) if lat else float("nan")
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--cache", default="samples/sweep.pkl")
    a = ap.parse_args()
    if os.path.exists(a.cache):
        cache = pickle.load(open(a.cache, "rb"))
        decoder = eng.Engine().decoder
    else:
        cache, decoder = collect(a.seqs, a.seed)
        pickle.dump(cache, open(a.cache, "wb"))
    print(f"{len(cache)} sequences")
    rows = []
    for ma, share, fc, win in itertools.product([2, 3, 4, 5], [0.5, 0.6, 0.75], [0.3, 0.5, 0.7], [8, 12]):
        r = evaluate(cache, decoder, {"MIN_AGREE": ma, "AGREE_SHARE": share, "FRAME_CONF": fc, "WINDOW": win})
        rows.append((r["correct"], -r["false_grant"], -r["wrong_events"], ma, share, fc, win, r))
    rows.sort(reverse=True)
    print(f"{'MIN_AGREE':>9}{'SHARE':>7}{'FRAME_CONF':>11}{'WINDOW':>7}{'correct':>9}{'false grant':>12}{'no decision':>12}{'wrong ev':>9}{'frames':>7}")
    for *_, ma, share, fc, win, r in rows[:15]:
        print(f"{ma:>9}{share:>7}{fc:>11}{win:>7}{r['correct']:>5}/{r['n']:<3}{r['false_grant']:>12}{r['no_decision']:>12}{r['wrong_events']:>9}{r['latency']:>7.0f}")
