"""Georgian licence plates: templates, constrained decoding, formatting.

The OCR model gives a probability for every character at every slot.
Instead of taking the argmax string and patching it (8->B, 0->O ...), we
decode directly against the template: a letter slot takes the most likely
*letter*, a digit slot the most likely *digit*, and the slots after the
plate must be padding. The plate's confidence is its weakest slot, so a
character the model was unsure about can never sneak through.

Templates: L = Latin letter, D = digit, "-" = separator in the display form.
"""
from __future__ import annotations

import itertools
import math
import os
import re
from dataclasses import dataclass

import numpy as np

# Standard plates since 2004: AB-123-CD. Others (legacy "ABC-123", trailers,
# special series) can be enabled with PLATE_TEMPLATES="LL-DDD-LL,LLL-DDD".
DEFAULT_TEMPLATES = "LL-DDD-LL"

MAX_STUTTER = int(os.environ.get("MAX_STUTTER", "1"))
STUTTER_LOGP = math.log(float(os.environ.get("STUTTER_PENALTY", "0.3")))

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"


@dataclass(frozen=True)
class Plate:
    plate: str                 # canonical, no separators: AB123CD
    display: str               # AB-123-CD
    template: str              # LL-DDD-LL
    slot_probs: tuple          # probability of each chosen char (plate slots, then padding)
    conf: float                # weakest slot
    logp: float                # sum of log-probs (template comparison)
    stutters: int = 0          # duplicated slots that were skipped


def templates() -> list[str]:
    raw = os.environ.get("PLATE_TEMPLATES", DEFAULT_TEMPLATES)
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _display(plate: str, template: str) -> str:
    out, i = [], 0
    for kind in template:
        if kind == "-":
            out.append("-")
        else:
            out.append(plate[i]); i += 1
    return "".join(out)


def _matches(s: str, pattern: str) -> bool:
    return len(s) == len(pattern) and all(
        (c in LETTERS if k == "L" else c in DIGITS) for c, k in zip(s, pattern))


class Decoder:
    def __init__(self, alphabet: str, pad_char: str) -> None:
        self.alphabet = alphabet
        self.letter_idx = np.array([alphabet.index(c) for c in LETTERS if c in alphabet])
        self.digit_idx = np.array([alphabet.index(c) for c in DIGITS if c in alphabet])
        self.pad_idx = alphabet.index(pad_char)

    def _stutter_sets(self, probs: np.ndarray, n: int, max_stutter: int):
        """Slot sets to drop: OCR models sometimes emit a character twice ("NO7177AH", "AHH").
        A slot may be dropped only if its argmax equals the previous slot's argmax."""
        am = probs.argmax(axis=1)
        cands = [i for i in range(1, probs.shape[0]) if am[i] == am[i - 1] and am[i] != self.pad_idx]
        yield ()
        for k in range(1, max_stutter + 1):
            yield from itertools.combinations(cands, k)

    def decode(self, probs: np.ndarray, only: str | None = None, max_stutter: int | None = None) -> Plate | None:
        """Best template-conforming plate for a (slots, alphabet) probability matrix.

        only: score exactly this canonical plate string (for a known candidate).
        """
        if max_stutter is None:
            max_stutter = MAX_STUTTER
        slots = probs.shape[0]
        best: Plate | None = None
        for tpl in templates():
            pattern = tpl.replace("-", "")
            n = len(pattern)
            if only is not None and not _matches(only, pattern):
                continue
            for drop in self._stutter_sets(probs, n, max_stutter):
                use = [i for i in range(slots) if i not in drop]
                if len(use) < n or any(d - 1 > use[n - 1] for d in drop):
                    continue  # a dropped slot must repeat a plate character, not padding
                chars, ps = [], []
                for kind, i in zip(pattern, use[:n]):
                    if only is not None:
                        j = self.alphabet.index(only[len(chars)])
                    else:
                        idx = self.letter_idx if kind == "L" else self.digit_idx
                        j = int(idx[probs[i, idx].argmax()])
                    chars.append(self.alphabet[j]); ps.append(float(probs[i, j]))
                ps += [float(probs[i, self.pad_idx]) for i in use[n:]]
                arr = np.clip(np.array(ps), 1e-9, 1.0)
                logp = float(np.log(arr).sum()) + len(drop) * STUTTER_LOGP
                text = "".join(chars)
                cand = Plate(text, _display(text, tpl), tpl, tuple(ps), float(arr.min()), logp, len(drop))
                if best is None or cand.logp > best.logp:
                    best = cand
        return best


def normalize_user_input(text: str) -> str | None:
    """Canonical form for a plate typed by a person; strict."""
    s = clean(text)
    for tpl in templates():
        if _matches(s, tpl.replace("-", "")):
            return s
    return None


def display(plate: str) -> str:
    for tpl in templates():
        if _matches(plate, tpl.replace("-", "")):
            return _display(plate, tpl)
    return plate
