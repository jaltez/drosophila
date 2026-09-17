#!/usr/bin/env python3
"""Vocabulary pin tests: a future vocab.py edit must not silently remap
values the pinned upstream datasets actually contain. Run directly
(python3 bridge/test_vocab.py) or under pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import vocab


def test_scale_is_kernel_contract():
    assert vocab.SCALE == 512


def test_sign_map():
    assert vocab.SIGN == {
        "acetylcholine": 1.0,
        "gaba": -1.0,
        "glutamate": -1.0,
        "histamine": -1.0,
        "dopamine": 0.5,
        "serotonin": 0.5,
        "octopamine": 0.5,
        "tyramine": 0.5,
    }


def test_func_code():
    assert vocab.FUNC_CODE == {"inter": 0, "sensory": 1, "dn": 2, "motor": 3}


def test_malecns_side_includes_M():
    # the 2026-09-17 fix: v1.0 writes midline as "M"; 392 neurons depend on it
    assert vocab.MALECNS_SIDE == {"L": 1, "R": 2, "M": 3, "midline": 3}


def test_flywire_side():
    assert vocab.SIDE == {"left": 1, "right": 2, "center": 3}
    assert "na" in vocab.SIDE_UNKNOWN


def test_keep_status_enumerated():
    assert vocab.KEEP_STATUS == frozenset({
        "Reviewed", "Roughly traced", "Prelim Roughly traced",
        "RT Hard to trace", "Hard to trace", "Partially traced",
        "0.5assign", "Anchor", "Soma Anchor", "Sensory Anchor",
    })


def test_side_classifier_m_row():
    classify = vocab.malecns_side_classifier(vocab.VocabReport("test"))
    assert classify("M", None) == 3
    assert classify("L", None) == 1
    assert classify(None, "R") == 2
    assert classify(None, None) == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
