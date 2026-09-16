#!/usr/bin/env python3
"""Compare male (MaleCNS) vs female (FlyWire) DN olfactory responses.

Inputs: runs/dn_screen_inst.json (male), runs/dn_screen_inst_female.json
(female) — per-instance rates under left-/right-antenna drive from the
identical stimulation protocol. Aggregates to type level (mean over
instances) and reports:
  - shared types with significant lateralized response in BOTH sexes
  - per-type response correlation between sexes (L-drive vector)
  - side-convention agreement: fraction of types whose preferred side
    (of responding instances) matches across sexes
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).parent / ".." / "runs"


def load_types(path):
    rows = json.load(open(path))
    by_type = defaultdict(lambda: {"L": [], "R": [], "base": [], "side": []})
    for r in rows:
        t = by_type[r["type"]]
        t["L"].append(r["left_hz"])
        t["R"].append(r["right_hz"])
        t["base"].append(r.get("base_hz", 0.0))
        t["side"].append(r["side"])
    return by_type


def type_agg(by_type):
    out = {}
    for t, d in by_type.items():
        L, R = np.mean(d["L"]), np.mean(d["R"])
        out[t] = {
            "left": L, "right": R, "base": float(np.mean(d["base"])),
            "asym": (L - R) / (L + R + 1e-9),
            "n": len(d["L"]),
        }
    return out


def main():
    male = type_agg(load_types(RUNS / "dn_screen_inst.json"))
    female = type_agg(load_types(RUNS / "dn_screen_inst_female.json"))
    shared = sorted(set(male) & set(female))
    print(f"male types: {len(male)}, female types: {len(female)}, shared: {len(shared)}")

    # response vectors under left drive, correlated across shared types
    m = np.array([male[t]["left"] for t in shared])
    f = np.array([female[t]["left"] for t in shared])
    if m.std() > 0 and f.std() > 0:
        corr = float(np.corrcoef(m, f)[0, 1])
    else:
        corr = 0.0
    print(f"corr(left-drive response, male vs female, shared types) = {corr:+.3f}")

    # lateralized responders in both sexes
    def lat(d):
        return abs(d["asym"]) > 0.25
    both_lat = [t for t in shared if lat(male[t]) and lat(female[t])]
    same_sign = [t for t in both_lat if np.sign(male[t]["asym"]) == np.sign(female[t]["asym"])]
    print(f"lateralized in both sexes: {len(both_lat)}; same preferred side: {len(same_sign)}")
    for t in both_lat[:15]:
        print(
            f"  {t:<10} male L{male[t]['left']:6.2f} R{male[t]['right']:6.2f} "
            f"asym {male[t]['asym']:+.2f} | female L{female[t]['left']:6.2f} "
            f"R{female[t]['right']:6.2f} asym {female[t]['asym']:+.2f}"
        )

    out = {
        "shared_types": len(shared),
        "left_drive_response_corr": corr,
        "lateralized_both": len(both_lat),
        "same_sign": len(same_sign),
        "rows": [
            {"type": t,
             "male": {k: male[t][k] for k in ("left", "right", "asym", "n")},
             "female": {k: female[t][k] for k in ("left", "right", "asym", "n")}}
            for t in both_lat
        ],
    }
    (RUNS / "sex_comparison.json").write_text(json.dumps(out, indent=1))
    print("wrote runs/sex_comparison.json")


if __name__ == "__main__":
    main()
