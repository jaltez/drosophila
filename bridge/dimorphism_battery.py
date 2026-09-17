#!/usr/bin/env python3
"""Functional dimorphism battery: male vs female DN response profiles.

For each DN type, the functional response vector is (base, left, right) Hz
from the per-instance screens (identical protocol both sexes). Functional
dimorphism index: FDI = ||m - f|| / (||m|| + ||f||).

Question: do types with high FDI enrich for anatomical dimorphism
(MaleCNS `dimorphism` column) and fruDsx expression (`fruDsx` column)?
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from analysis import enrichment_z, fdi
from vocab import KEEP_STATUS

RUNS = Path(__file__).parent / ".." / "runs"
ANNOT = Path(__file__).parent / "data" / (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather")


def load_types(path):
    rows = json.load(open(path))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["type"]]["base"].append(r.get("base_hz", 0.0))
        by[r["type"]]["L"].append(r["left_hz"])
        by[r["type"]]["R"].append(r["right_hz"])
    return {t: (np.mean(d["base"]), np.mean(d["L"]), np.mean(d["R"]))
            for t, d in by.items()}





def main():
    male = load_types(RUNS / "dn_screen_inst.json")
    female = load_types(RUNS / "dn_screen_inst_female.json")
    shared = sorted(set(male) & set(female))

    # per-sex gain normalization: scale each sex's response vectors by the
    # grand mean over shared types, so FDI reflects response-profile SHAPE
    # rather than overall excitability differences between calibrations
    for d in (male, female):
        g = np.mean([sum(v) / len(v) for v in d.values()])
        for t in d:
            d[t] = tuple(x / g for x in d[t])

    rows = []
    censored = 0
    for t in shared:
        mtot = sum(male[t])
        ftot = sum(female[t])
        # threshold censoring: the female brain runs colder under its own
        # homeostatic calibration; types silent in either sex cannot have
        # their response SHAPE compared, only their silence noted
        if mtot < 0.5 or ftot < 0.5:
            censored += 1
            continue
        rows.append({"type": t, "fdi": fdi(male[t], female[t]),
                     "male_L": male[t][1], "male_R": male[t][2],
                     "female_L": female[t][1], "female_R": female[t][2]})
    fdi_arr = np.array([r["fdi"] for r in rows])
    print(f"shared types: {len(shared)}; measurable in both sexes: "
          f"{len(rows)}; threshold-censored: {censored}")

    # anatomical flags from MaleCNS annotations (type-level: any instance)
    an = feather.read_table(
        ANNOT,
        columns=["bodyId", "type", "dimorphism", "fruDsx", "statusLabel"],
    ).to_pandas()
    an = an[an.statusLabel.isin(KEEP_STATUS)]
    dim_types = set(an.loc[an.dimorphism.notna(), "type"].dropna())
    frutypes = set(an.loc[an.fruDsx.notna(), "type"].dropna())
    print(f"annotated: {len(dim_types)} dimorphism-flagged types, "
          f"{len(frutypes)} fruDsx-flagged types")

    for r in rows:
        r["anat_dimorphic"] = r["type"] in dim_types
        r["fru"] = r["type"] in frutypes

    flagged = np.array([r["anat_dimorphic"] for r in rows])
    fru = np.array([r["fru"] for r in rows])

    z_dim, _ = enrichment_z(fdi_arr, flagged, "anatomical dimorphism")
    z_fru, _ = enrichment_z(fdi_arr, fru, "fruDsx")

    rows.sort(key=lambda r: -r["fdi"])
    print("\ntop-15 functionally dimorphic DN types:")
    print(f"{'type':<10} {'FDI':>6} {'anat':>5} {'fru':>4} "
          f"{'male L/R':>12} {'female L/R':>13}")
    for r in rows[:15]:
        print(f"{r['type']:<10} {r['fdi']:>6.2f} {str(r['anat_dimorphic']):>5} "
              f"{str(r['fru']):>4} {r['male_L']:5.1f}/{r['male_R']:<5.1f} "
              f"{r['female_L']:6.1f}/{r['female_R']:<5.1f}")

    out = {
        "shared_types": len(shared),
        "fdi_mean": float(fdi_arr.mean()),
        "enrichment": {"anatomical_dimorphism_z": z_dim,
                       "frudsx_z": z_fru},
        "dimorphism_flagged_types": len(dim_types),
        "frudsx_flagged_types": len(frutypes),
        "rows": rows,
    }
    (RUNS / "functional_dimorphism.json").write_text(json.dumps(out, indent=1))
    print("\nwrote runs/functional_dimorphism.json")


if __name__ == "__main__":
    main()
