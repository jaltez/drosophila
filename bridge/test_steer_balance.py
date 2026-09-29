#!/usr/bin/env python3
"""Steering-set balance pins: the checked-in chemotactic steering instance
files must be mass-balanced (sum of positive w == sum of |negative w|),
otherwise common-mode rate excursions leak into the steering sum with
gain (mass_pos - mass_neg) and saturate the turn-rate command. Run
directly (python3 bridge/test_steer_balance.py) or under pytest.

Visual steering sets are exempt: their gain structure is the validated
optomotor open-loop transfer, and in the chemotaxis arena the visual
channel idles at a constant static rate (no common-mode fluctuations).
"""
import json
import sys
from pathlib import Path

SCEN = Path(__file__).parent.parent / "scenarios"

CHEMO_SETS = [
    "steer_instances.json",
    "steer_instances_female.json",
    "steer_instances_opponent.json",
]

TOLERANCE = 0.02  # relative mass imbalance tolerated (rounding at 4 decimals)


def _masses(path):
    insts = json.load(open(path))
    pos = sum(i["w"] for i in insts if i["w"] > 0)
    neg = sum(-i["w"] for i in insts if i["w"] < 0)
    return pos, neg


def test_chemo_sets_are_mass_balanced():
    for name in CHEMO_SETS:
        p = SCEN / name
        if not p.exists():
            continue
        pos, neg = _masses(p)
        total = pos + neg
        assert total > 0, f"{name}: zero authority"
        assert abs(pos - neg) / total <= TOLERANCE, (
            f"{name}: mass_pos {pos:.1f} vs mass_neg {neg:.1f} "
            f"(common-mode gain {pos - neg:+.1f}); run balance_steer_weights.py"
        )


if __name__ == "__main__":
    try:
        test_chemo_sets_are_mass_balanced()
        print("ok: chemotactic steering sets mass-balanced")
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
