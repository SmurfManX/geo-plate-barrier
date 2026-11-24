"""Single-frame OCR accuracy on real crops, several configs.

python -m tests.real_eval samples/real
The folder holds plate crops (*.jpg, e.g. exported from the access log) and
labels.csv with lines "filename,PLATE" (PLATE without dashes).
"""
import glob
import os
import sys

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.inference import plate_recognizer as fpr

from app import plates


def load_labels(path: str) -> dict[str, str]:
    out = {}
    with open(os.path.join(path, "labels.csv")) as f:
        for line in f:
            if "," in line:
                name, plate = line.strip().split(",", 1)
                out[name] = plates.clean(plate)
    return out


def clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma(bgr, g=2.0):
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def autolevel(bgr):
    lo, hi = np.percentile(bgr, (2, 98))
    return np.clip((bgr.astype(np.float32) - lo) * 255 / max(1, hi - lo), 0, 255).astype(np.uint8)


PRE = {"none": lambda x: x, "clahe": clahe, "gamma2": gamma, "autolevel": autolevel}


class M:
    def __init__(self, name):
        self.r = LicensePlateRecognizer(name, device="cpu")
        c = self.r.config
        self.alphabet, self.pad = c.alphabet, c.pad_char
        self.rgb = c.image_color_mode == "rgb"
        self.dec = plates.Decoder(self.alphabet, self.pad)

    def probs(self, bgr_list):
        imgs = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB if self.rgb else cv2.COLOR_BGR2GRAY) for b in bgr_list]
        x = fpr.preprocess_image(fpr._load_image_from_source(imgs, self.r.config))
        out = self.r.model.run([self.r.plate_output_name], {"input": x})[0]
        return out.reshape(len(bgr_list), self.r.config.max_plate_slots, -1).astype(np.float64)


def geo(ps):
    lp = np.mean([np.log(np.clip(p, 1e-9, 1)) for p in ps], axis=0)
    p = np.exp(lp - lp.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)


def main(path):
    labels = load_labels(path)
    data = [(cv2.imread(os.path.join(path, n)), t) for n, t in sorted(labels.items())]
    print(f"{len(data)} labelled crops")
    models = {}
    for name in ["cct-s-v2-global-model", "cct-xs-v2-global-model", "cct-s-v1-global-model",
                 "global-plates-mobile-vit-v2-model", "european-plates-mobile-vit-v2-model"]:
        try:
            models[name] = M(name)
        except Exception as e:
            print("skip", name, e)
    # cache probs per model/preproc
    P = {}
    for mn, m in models.items():
        for pn, fn in PRE.items():
            P[(mn, pn)] = m.probs([fn(img) for img, _ in data])
    truths = [t for _, t in data]

    def score(name, prob_list, dec, frame_conf=0.5, stutter=1):
        ok = wrong = 0; wrongs = []
        for p, t in zip(prob_list, truths):
            d = dec.decode(p, max_stutter=stutter)
            if d and d.conf >= frame_conf:
                if d.plate == t: ok += 1
                else: wrong += 1; wrongs.append(f"{t}->{d.plate}({d.conf:.2f})")
        print(f"  {name:<58} ok {ok:>2}/{len(truths)}  wrong {wrong:>2}  rejected {len(truths)-ok-wrong:>2}   {' '.join(wrongs[:4])}")
        return ok, wrong

    base = models["cct-s-v2-global-model"].dec
    print("\n-- single model, single preprocessing (stutter=1, frame_conf=0.5)")
    for (mn, pn), pr in P.items():
        if models[mn].alphabet == models["cct-s-v2-global-model"].alphabet and pr.shape[1] == 10:
            score(f"{mn} / {pn}", pr, models[mn].dec)
        else:
            score(f"{mn} / {pn} (own decoder)", pr, models[mn].dec)
    print("\n-- stutter handling (cct-s-v2 / none)")
    for st in (0, 1, 2):
        score(f"max_stutter={st}", P[("cct-s-v2-global-model", "none")], base, stutter=st)
    print("\n-- test-time augmentation: geometric mean over preprocessings (cct-s-v2)")
    for combo in (["none", "clahe"], ["none", "gamma2"], ["none", "autolevel"], ["none", "clahe", "gamma2", "autolevel"]):
        pr = [geo([P[("cct-s-v2-global-model", c)][i] for c in combo]) for i in range(len(truths))]
        score("+".join(combo), pr, base)
    print("\n-- model ensembles (same alphabet & slots), preprocessing none")
    same = [mn for mn in models if models[mn].alphabet == base.alphabet and P[(mn, "none")].shape[1:] == P[("cct-s-v2-global-model", "none")].shape[1:]]
    print("  compatible:", same)
    import itertools
    for k in (2, 3):
        for combo in itertools.combinations(same, k):
            pr = [geo([P[(mn, "none")][i] for mn in combo]) for i in range(len(truths))]
            score(" + ".join(c.replace("-global-model", "").replace("-model", "") for c in combo), pr, base)
    for combo in (["cct-s-v2-global-model", "global-plates-mobile-vit-v2-model"],):
        if all(c in same for c in combo):
            pr = [geo([P[(mn, pn)][i] for mn in combo for pn in ("none", "clahe")]) for i in range(len(truths))]
            score("cct-s-v2+mobilevit x (none,clahe)", pr, base)


if __name__ == "__main__":
    main(sys.argv[1])
