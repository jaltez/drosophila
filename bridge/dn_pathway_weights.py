#!/usr/bin/env python3
"""Anatomical steering weights: propagate left/right olfactory ORN activity
through the connectome graph (3 hops), score each DN instance by
L-vs-R drive, and write scenarios/steer_instances.json for the arbiter.

Antisymmetric by construction of the wiring: a DN driven by the left
olfactory pathway gets positive weight, one driven by the right pathway
negative. Baselines come from the per-instance dn_screen (no-odor rates).
"""
import json
import sys
from pathlib import Path

import numpy as np

SUP = sys.argv[1] if len(sys.argv) > 1 else "cb_sensory"
OUT = sys.argv[2] if len(sys.argv) > 2 else "scenarios/steer_instances.json"
ART = sys.argv[3] if len(sys.argv) > 3 else "out/connectome-malecns-v1.0"
SCREEN = sys.argv[4] if len(sys.argv) > 4 else "runs/dn_screen_inst.json"
D = Path(__file__).parent / ART

fc = np.fromfile(D / "func_classes.u8", dtype=np.uint8)
sc = np.fromfile(D / "superclass_codes.u8", dtype=np.uint8)
sides = np.fromfile(D / "side_codes.u8", dtype=np.uint8)
sup_names = json.load(open(D / "superclass_names.json"))
cb_sensory = sup_names.index(SUP)

n = int(np.fromfile(D / "body_ids.u64", dtype=np.uint64).shape[0])
offsets = np.fromfile(D / "csr_offsets.u64", dtype=np.uint64)
targets = np.fromfile(D / "csr_targets.u32", dtype=np.uint32)
weights = np.fromfile(D / "csr_weights.i16", dtype=np.int16)

# symmetric excitatory-ish connectivity: |w| both directions
src = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets).astype(np.int64))
w_abs = np.abs(weights).astype(np.float64)

# column-normalize by target in-strength (postsynaptic normalization)
in_str = np.zeros(n)
np.add.at(in_str, targets, w_abs)
in_str[in_str == 0] = 1.0
wnorm = w_abs / in_str[targets]

def diffuse(seed: np.ndarray, hops: int = 3, retain: float = 0.35) -> np.ndarray:
    x = seed.astype(np.float64)
    for _ in range(hops):
        spread = np.zeros(n)
        np.add.at(spread, targets, wnorm * x[src])
        x = retain * seed + (1.0 - retain) * spread
    return x

orn_l = np.where((fc == 1) & (sc == cb_sensory) & (sides == 1))[0]
orn_r = np.where((fc == 1) & (sc == cb_sensory) & (sides == 2))[0]
print(f"seeds: {len(orn_l)} left ORNs, {len(orn_r)} right ORNs")

seed_l = np.zeros(n)
seed_l[orn_l] = 1.0
seed_r = np.zeros(n)
seed_r[orn_r] = 1.0

xl = diffuse(seed_l)
xr = diffuse(seed_r)

dn = np.where(fc == 2)[0]
base_by_body = {}
screen_path = Path(__file__).parent / ".." / SCREEN
for r in json.load(open(screen_path)):
    base_by_body[r["bodyId"]] = r.get("base_hz", 0.0)

insts = []
for i in dn:
    s = xl[i] - xr[i]
    tot = xl[i] + xr[i]
    if tot < 1e-9:
        continue
    w = s / tot  # -1..1, antisymmetric by construction
    if abs(w) < 0.05:
        continue
    body = int(np.fromfile(D / "body_ids.u64", dtype=np.uint64, count=1, offset=i * 8)[0])
    insts.append({
        "bodyId": body,
        "w": round(float(w), 3),
        "base": round(base_by_body.get(body, 0.0), 3),
    })

insts.sort(key=lambda d: -abs(d["w"]))
pos = sum(1 for d in insts if d["w"] > 0)
out = Path(__file__).parent / ".." / OUT
json.dump(insts, open(out, "w"), indent=1)
print(f"{len(insts)} DN steering instances ({pos} left-driven, {len(insts)-pos} right-driven)")
print("top:", [(d['bodyId'], d['w']) for d in insts[:5]])
print(f"wrote {out}")
