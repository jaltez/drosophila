#!/usr/bin/env python3
"""Opponent steering-weight construction.

Instead of diffusion-derived weights (dn_olfactory_weights.py), build each
DN instance's steering weight from its MEASURED unilateral-odor response
asymmetry (dn_screen per-instance data):

    w_i = (rL_i - rR_i) / (rL_i + rR_i)      for instances with rL+rR >= 1 Hz
    base_i = base_hz_i                        (screen baseline; in-assay
                                              centering refines it anyway)

The set is rescaled to match the old construction's total drive authority
(sum |w|), so a head-to-head lock-on assay isolates CONSTRUCTION — opponent
sign from measured responses — as the only change.

usage: dn_opponent_weights.py [screen.json] [out.json]
"""
import json
import sys

import numpy as np

screen_path = sys.argv[1] if len(sys.argv) > 1 else "runs/dn_screen_inst.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else "scenarios/steer_instances_opponent.json"
old_path = "scenarios/steer_instances.json"

rows = json.load(open(screen_path))
old = json.load(open(old_path))
authority = sum(abs(i["w"]) for i in old)

rL = np.array([r["left_hz"] for r in rows])
rR = np.array([r["right_hz"] for r in rows])
mag = rL + rR
keep = mag >= 1.0

asym = np.where(keep, (rL - rR) / np.maximum(mag, 1e-9), 0.0)
raw_authority = float(np.abs(asym).sum())
scale = authority / max(raw_authority, 1e-9)

insts = [
    {"bodyId": r["bodyId"], "w": float(round(a * scale, 4)), "base": r["base_hz"]}
    for r, a in zip(rows, asym)
    if keep[np.array([x is r for x in rows]).argmax()] and abs(a) > 1e-6
]
# (rebuild cleanly: keep rows where keep[i] and asym nonzero)
insts = []
for r, a, k in zip(rows, asym, keep):
    if k and abs(a) > 1e-6:
        insts.append({"bodyId": r["bodyId"], "w": round(float(a * scale), 4),
                      "base": r["base_hz"]})

json.dump(insts, open(out_path, "w"))
print(f"opponent set: {len(insts)} instances, sum|w| {sum(abs(i['w']) for i in insts):.1f} "
      f"(authority-matched to old {authority:.1f}), scale {scale:.2f}")
