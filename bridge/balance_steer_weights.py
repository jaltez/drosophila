#!/usr/bin/env python3
"""Mass-balance a steering-instance weight set.

The steering drive is sum(w_i * (rate_i - base_i)) with + = prefers
left-antenna drive. When the positive-side mass (sum of w over
left-preferring instances) differs from the negative-side mass, every
COMMON-MODE rate excursion (odor-driven global rate changes, kernel
noise shared across the population) leaks into the steering sum with
gain (mass_pos - mass_neg). Measured in the male chemotaxis assay:
mass_pos 620 vs mass_neg 273 -> ~347x common-mode gain, steering drive
noise sigma ~116 Hz against a chemotactic signal of ~4 Hz mean — the
turn-rate command sat in tanh bang-bang saturation and the fly pivoted
in place instead of translating.

This script rescales each sign group so both masses equal half the
total |w| authority (relative weights within a group are preserved, the
measured sign structure is untouched). Construction scripts stay
authoritative for the signs; balancing is a calibration step applied
afterwards, and the result is pinned by test_steer_balance.py.

usage: balance_steer_weights.py <instances.json> [out.json]
       (in-place rewrite when out.json is omitted)
"""
import json
import sys

path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else path

insts = json.load(open(path))
pos = [i for i in insts if i["w"] > 0]
neg = [i for i in insts if i["w"] < 0]
mass_pos = sum(i["w"] for i in pos)
mass_neg = sum(-i["w"] for i in neg)

if not pos or not neg:
    sys.exit(f"{path}: one sign group empty (pos {len(pos)}, neg {len(neg)}), cannot balance")

total = mass_pos + mass_neg
target = total / 2.0
k_pos = target / mass_pos
k_neg = target / mass_neg

for i in pos:
    i["w"] = round(i["w"] * k_pos, 4)
for i in neg:
    i["w"] = round(i["w"] * k_neg, 4)

json.dump(insts, open(out_path, "w"))
print(f"{path}: mass_pos {mass_pos:.1f} mass_neg {mass_neg:.1f} "
      f"(common-mode gain {mass_pos - mass_neg:+.1f}) -> "
      f"mass {target:.1f}/{target:.1f} (gain ~0), {len(pos)}+/{len(neg)}- instances")
