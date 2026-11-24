import numpy as np

from app import plates

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
DEC = plates.Decoder(ALPHABET, "_")


def probs(text: str, slots: int = 10, sure: float = 0.97, alt: dict | None = None) -> np.ndarray:
    """Probability matrix that reads `text`; alt={slot: (char, p)} adds a competing char."""
    text = text.ljust(slots, "_")
    p = np.full((slots, len(ALPHABET)), (1 - sure) / (len(ALPHABET) - 1))
    for i, ch in enumerate(text):
        p[i, ALPHABET.index(ch)] = sure
    for i, (ch, q) in (alt or {}).items():
        small = 0.001
        p[i, :] = small
        p[i, ALPHABET.index(ch)] = q
        p[i, ALPHABET.index(text[i])] = 1 - q - small * (len(ALPHABET) - 2)
    return p / p.sum(axis=1, keepdims=True)


def test_clean_read():
    d = DEC.decode(probs("KA505BA"))
    assert d.plate == "KA505BA" and d.display == "KA-505-BA" and d.conf > 0.9


def test_digit_in_letter_slot_takes_best_letter_with_low_conf():
    # OCR says "8" in a letter slot: decoder picks the best *letter*, but confidence collapses
    d = DEC.decode(probs("8U100LI"))
    assert d.plate[1:] == "U100LI" and d.plate[0].isalpha()
    assert d.conf < 0.1


def test_competing_letter_decides_by_probability():
    d = DEC.decode(probs("BU100LI", alt={0: ("8", 0.6)}))   # "8" 0.6 vs "B" ~0.37 at slot 0
    assert d.plate == "BU100LI" and d.conf < 0.5


def test_extra_char_must_be_padding():
    d = DEC.decode(probs("NO717AHX"))
    assert d.plate == "NO717AH" and d.conf < 0.1


def test_stutter_is_skipped():
    d = DEC.decode(probs("NO7177AH"))
    assert d.plate == "NO717AH" and d.stutters == 1 and d.conf > 0.9
    d = DEC.decode(probs("NO717AHH"))
    assert d.plate == "NO717AH" and d.stutters == 1


def test_score_known_candidate():
    d = DEC.decode(probs("NO717AH"), only="MO717AH")
    assert d.plate == "MO717AH" and d.conf < 0.05


def test_display_and_user_input():
    assert plates.display("BU100LI") == "BU-100-LI"
    assert plates.normalize_user_input(" bu 100 li ") == "BU100LI"
    assert plates.normalize_user_input("BU-1OO-LI") is None
    assert plates.normalize_user_input("ABC-123") is None      # legacy format off by default
