#!/usr/bin/env python3
"""DN stimulus-battery analysis: 6-protocol response profiles (base,
odor-L/R, vis-L/R, sugar) from `dn_battery` runs on both calibrated brains.

Questions:
1. Sex conservation: do per-type response magnitudes agree across sexes?
   (Pearson r over shared types, per protocol and pooled)
2. Functional dimorphism from the battery: FDI over the 5-stimulus
   response vector (base excluded), per-sex gain normalization and
   threshold censoring as in dimorphism_battery.py.
3. Enrichment: do high-FDI types enrich for MaleCNS anatomical
   dimorphism / fruDsx flags?
4. Lateralization: types with significant |L-R| odor/visual asymmetry in
   BOTH sexes (side-sign agreement noted).

Writes runs/dn_profile_battery.json.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

RUNS = Path(__file__).parent / ".." / "runs"
ANNOT = Path(__file__).parent / "data" / (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather")

PROTOS = ["odor_L", "odor_R", "vis_L", "vis_R", "sugar"]


def load_battery(path):
    rows = json.load(open(path))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d = by[r["type"]]
        for p in ["base"] + PROTOS:
            d[p].append(r.get(p, 0.0))
    # per-type mean across instances
    out = {}
    for t, d in by.items():
        out[t] = {p: float(np.mean(d[p])) for p in ["base"] + PROTOS}
    return out


def resp_vec(d):
    """Stimulus-evoked response vector: protocol rate minus base, floored
    at 0 (a protocol that silences a type below baseline carries no
    positive response magnitude)."""
    return np.array([max(d[p] - d["base"], 0.0) for p in PROTOS])


def fdi(m, f):
    dm, df = np.asarray(m, float), np.asarray(f, float)
    return float(np.linalg.norm(dm - df) /
                 (np.linalg.norm(dm) + np.linalg.norm(df) + 1e-9))


def main():
    male = load_battery(RUNS / "dn_battery_male.json")
    female = load_battery(RUNS / "dn_battery_female.json")
    shared = sorted(set(male) & set(female))
    print(f"male types: {len(male)}, female types: {len(female)}, "
          f"shared: {len(shared)}")

    # per-sex gain normalization (grand mean over shared types, all
    # protocols + base): FDI reflects profile shape, not excitability
    for d in (male, female):
        g = np.mean([sum(v.values()) / len(v) for v in d.values()])
        for t in d:
            d[t] = {p: v / g for p, v in d[t].items()}

    # 1. conservation: per-protocol Pearson of evoked magnitude across
    #    shared types, pooled over protocols
    cons = {}
    pm, pf = [], []
    for p in PROTOS:
        m = [max(male[t][p] - male[t]["base"], 0.0) for t in shared]
        f = [max(female[t][p] - female[t]["base"], 0.0) for t in shared]
        r = float(np.corrcoef(m, f)[0, 1])
        cons[p] = r
        pm.extend(m)
        pf.extend(f)
    r_pooled = float(np.corrcoef(pm, pf)[0, 1]) if pm else float("nan")
    print("conservation r per protocol:",
          {k: round(v, 3) for k, v in cons.items()},
          f"pooled r={r_pooled:+.3f}")

    # 2. FDI over battery profiles with threshold censoring
    rows = []
    censored = 0
    for t in shared:
        mtot = sum(male[t].values())
        ftot = sum(female[t].values())
        if mtot < 0.5 or ftot < 0.5:
            censored += 1
            continue
        rows.append({"type": t, "fdi": fdi(resp_vec(male[t]),
                                           resp_vec(female[t]))})
    fdi_arr = np.array([r["fdi"] for r in rows])
    print(f"measurable in both sexes: {len(rows)}; censored: {censored}; "
          f"FDI mean {fdi_arr.mean():.3f}")

    # 3. enrichment vs anatomical flags
    an = feather.read_table(
        ANNOT, columns=["bodyId", "type", "dimorphism", "fruDsx",
                        "statusLabel"]).to_pandas()
    keep = {"Reviewed", "Roughly traced", "Prelim Roughly traced",
            "RT Hard to trace", "Hard to trace", "Partially traced",
            "0.5assign", "Anchor", "Soma Anchor", "Sensory Anchor"}
    an = an[an.statusLabel.isin(keep)]
    dim_types = set(an.loc[an.dimorphism.notna(), "type"].dropna())
    fru_types = set(an.loc[an.fruDsx.notna(), "type"].dropna())
    for r in rows:
        r["anat_dimorphic"] = bool(r["type"] in dim_types)
        r["fru"] = bool(r["type"] in fru_types)
    flagged = np.array([r["anat_dimorphic"] for r in rows])
    fru = np.array([r["fru"] for r in rows])

    def enrich(mask, label):
        hi = fdi_arr >= np.quantile(fdi_arr, 0.75)
        a = int((hi & mask).sum())
        b = int((hi & ~mask).sum())
        c = int((~hi & mask).sum())
        dd = int((~hi & ~mask).sum())
        p_hi = a / max(a + b, 1)
        p_lo = c / max(c + dd, 1)
        p_pool = (a + c) / max(a + b + c + dd, 1)
        se = math.sqrt(p_pool * (1 - p_pool) *
                       (1 / max(a + b, 1) + 1 / max(c + dd, 1))) or 1e-9
        z = (p_hi - p_lo) / se
        print(f"{label:<24} top-FDI flagged: {a}/{a+b} ({p_hi:.0%}) vs "
              f"{c}/{c+dd} ({p_lo:.0%})  z={z:+.2f}")
        return z

    z_dim = enrich(flagged, "anatomical dimorphism")
    z_fru = enrich(fru, "fruDsx")

    # 4. lateralization in both sexes: odor |L-R| and vis |L-R|
    def lat(d, t, pa, pb):
        return abs(d[t][pa] - d[t][pb])

    both_lat = []
    for t in shared:
        m_odor = lat(male, t, "odor_L", "odor_R")
        f_odor = lat(female, t, "odor_L", "odor_R")
        m_vis = lat(male, t, "vis_L", "vis_R")
        f_vis = lat(female, t, "vis_L", "vis_R")
        # conservative: asymmetry must exceed 25% of the type's own
        # evoked magnitude in that sex (not mere noise)
        m_mag = max(male[t]["odor_L"], male[t]["odor_R"], 0.05)
        f_mag = max(female[t]["odor_L"], female[t]["odor_R"], 0.05)
        m_vm = max(male[t]["vis_L"], male[t]["vis_R"], 0.05)
        f_vm = max(female[t]["vis_L"], female[t]["vis_R"], 0.05)
        odor_lat = m_odor > 0.25 * m_mag and f_odor > 0.25 * f_mag
        vis_lat = m_vis > 0.25 * m_vm and f_vis > 0.25 * f_vm
        if odor_lat or vis_lat:
            both_lat.append({"type": t, "odor": bool(odor_lat),
                             "visual": bool(vis_lat),
                             "fdi": next((r["fdi"] for r in rows
                                          if r["type"] == t), None)})
    print(f"lateralized in BOTH sexes: {len(both_lat)} types")

    rows.sort(key=lambda r: -r["fdi"])
    print("\ntop-12 functionally dimorphic (battery FDI):")
    print(f"{'type':<12} {'FDI':>6} {'anat':>5} {'fru':>4}")
    for r in rows[:12]:
        print(f"{r['type']:<12} {r['fdi']:>6.2f} "
              f"{str(r['anat_dimorphic']):>5} {str(r['fru']):>4}")

    out = {
        "protocols": PROTOS,
        "shared_types": len(shared),
        "measurable": len(rows),
        "censored": censored,
        "conservation_r": {"per_protocol": {k: float(v) for k, v in cons.items()},
                           "pooled": r_pooled},
        "fdi_mean": float(fdi_arr.mean()),
        "enrichment": {"anatomical_dimorphism_z": float(z_dim),
                       "frudsx_z": float(z_fru),
                       "dimorphism_flagged_types": len(dim_types),
                       "frudsx_flagged_types": len(fru_types)},
        "lateralized_both_sexes": both_lat,
        "top_fdi": rows[:25],
    }
    (RUNS / "dn_profile_battery.json").write_text(json.dumps(out, indent=1))
    print("\nwrote runs/dn_profile_battery.json")


if __name__ == "__main__":
    main()
