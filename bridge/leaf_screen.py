#!/usr/bin/env python3
"""Leaf-level screen for held locomotion-family DN types.

The behavior-map curation report holds every review entry below the 0.90
seed-confidence band: blind validation reproduced only 1/5 there, and
promotion waits on LEAF evidence (walk vs walk_back) that the L/R
sensory screens cannot provide. This screen asks whether the connectome
itself separates the confident leaf references:

    forward walk : DNa01, DNa02  (Namiki et al. 2018, via behavior_map)
    backward     : MDN           (Bidaye et al. 2015, 'established')
    escape outgroup: DNp01 (giant fiber)

Features per type (aggregated over instances, |weight|-weighted):
  - output-partner superclass composition
  - output-partner centroid along each raw coordinate axis (the AP axis
    is selected empirically as the coordinate best separating brain-core
    (cb_intrinsic) from VNC-core (vnc_intrinsic) neurons)

Decision policy (fixed BEFORE looking at held types):
  - The feature space is leaf-informative only if the references
    separate: d(forward, MDN) > 2x mean within-forward distance
    (leave-one-out over instances).
  - Held locomotion types are then REPORTED by nearest reference with
    the distance ratio; nothing is auto-promoted. Promotion stays a
    curator decision per the curation policy — this script's job is to
    say whether leaf evidence EXISTS and for which types.

usage: leaf_screen.py [artifact_dir] [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).parent / (
    sys.argv[1] if len(sys.argv) > 1 else "out/connectome-malecns-v1.0")
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/leaf_screen.json"

FORWARD = ["DNa01", "DNa02"]
BACKWARD = ["MDN"]
OUTGROUP = ["DNp01"]


def load():
    ids = np.fromfile(ART / "body_ids.u64", dtype=np.uint64)
    off = np.fromfile(ART / "csr_offsets.u64", dtype=np.uint64)
    tgt = np.fromfile(ART / "csr_targets.u32", dtype=np.uint32)
    w = np.fromfile(ART / "csr_weights.i16", dtype=np.int16)
    pos = np.fromfile(ART / "positions.f32", dtype=np.float32).reshape(-1, 3)
    df = pd.read_parquet(ART / "neurons.parquet")
    sup_names = json.load(open(ART / "superclass_names.json"))
    sup = np.fromfile(ART / "superclass_codes.u8", dtype=np.uint8)
    return ids, off, tgt, w, pos, df, sup, sup_names


def ap_axis(pos, sup, sup_names):
    """Coordinate best separating brain-core from VNC-core neurons."""
    cb = pos[sup == sup_names.index("cb_intrinsic")].mean(0)
    vnc = pos[sup == sup_names.index("vnc_intrinsic")].mean(0)
    gap = np.abs(vnc - cb)
    axis = int(gap.argmax())
    return axis, {
        "cb_intrinsic_centroid": cb.tolist(),
        "vnc_intrinsic_centroid": vnc.tolist(),
        "selected_axis": axis,
        "gap_per_axis": gap.tolist(),
    }


def type_features(df, ids, off, tgt, w, pos, sup, sup_names, axis, types):
    """Per-type partner-composition + partner-centroid features."""
    row_of = {b: i for i, b in enumerate(ids.tolist())}
    feats = {}
    for t in types:
        rows = df[df["type"] == t]
        inst = []
        for bid in rows["bodyId"].values:
            i = row_of.get(int(bid))
            if i is None:
                continue
            s, e = int(off[i]), int(off[i + 1])
            if e <= s:
                continue
            tw = tgt[s:e]
            ww = np.abs(w[s:e]).astype(np.float64)
            tidx = ids.searchsorted(tw)
            ok = tidx < len(ids)
            tidx = tidx[ok]
            ww = ww[ok]
            pp = pos[tidx]
            comp = np.zeros(len(sup_names))
            for c in sup[tidx]:
                comp[c] += 1.0
            comp /= max(len(tw), 1)
            inst.append({
                "outsyn": int(e - s),
                "partner_centroid_axis": float((pp[:, axis] * ww).sum() / ww.sum()),
                "partner_superclass_frac": comp.tolist(),
            })
        if inst:
            feats[t] = {
                "n_instances": len(inst),
                "outsyn_mean": float(np.mean([x["outsyn"] for x in inst])),
                "partner_centroid_axis_mean": float(
                    np.mean([x["partner_centroid_axis"] for x in inst])),
                "partner_centroid_axis_std": float(
                    np.std([x["partner_centroid_axis"] for x in inst])),
                "partner_superclass_frac_mean": np.mean(
                    [x["partner_superclass_frac"] for x in inst], axis=0).tolist(),
            }
    return feats


def main():
    ids, off, tgt, w, pos, df, sup, sup_names = load()
    axis, axis_info = ap_axis(pos, sup, sup_names)
    dn_types = sorted(
        t for t in df[df["func_class"] == "dn"]["type"].unique() if t)

    feats = type_features(df, ids, off, tgt, w, pos, sup, sup_names, axis, dn_types)

    fwd = [t for t in FORWARD if t in feats]
    bwd = [t for t in BACKWARD if t in feats]
    out = {"axis": axis_info, "references": {}, "separation": None,
           "leaf_informative": None, "held_classification": []}

    for t in FORWARD + BACKWARD + OUTGROUP:
        if t in feats:
            out["references"][t] = feats[t]

    # separation: cross-forward distance vs forward->MDN distance
    if len(fwd) == len(FORWARD) and len(bwd) == len(BACKWARD):
        within = abs(feats[FORWARD[0]]["partner_centroid_axis_mean"]
                     - feats[FORWARD[1]]["partner_centroid_axis_mean"])
        cross = abs(np.mean([feats[t]["partner_centroid_axis_mean"] for t in fwd])
                    - feats[BACKWARD[0]]["partner_centroid_axis_mean"])
        out["separation"] = {
            "within_forward_axis_dist": within,
            "forward_to_mdn_axis_dist": cross,
            "ratio": cross / max(within, 1e-9),
        }
        out["leaf_informative"] = bool(cross > 2.0 * max(within, 1e-9))

    # held locomotion types: review entries the curation report holds
    seed_path = Path(__file__).parent.parent / "scenarios/behavior_map_seed.json"
    seed = json.load(open(seed_path))
    held = [k for k, v in seed.get("review", {}).items()
            if v.get("family_hint") == "locomotion"]
    for t in held:
        if t not in feats:
            continue
        x = feats[t]["partner_centroid_axis_mean"]
        d_fwd = min(abs(x - feats[r]["partner_centroid_axis_mean"]) for r in fwd)
        d_bwd = abs(x - feats[BACKWARD[0]]["partner_centroid_axis_mean"])
        out["held_classification"].append({
            "type": t,
            "nearest_leaf": "walk_forward" if d_fwd < d_bwd else "walk_back",
            "dist_to_forward": d_fwd,
            "dist_to_mdn": d_bwd,
            "confidence_axis_std": feats[t]["partner_centroid_axis_std"],
        })

    out["policy"] = (
        "leaf_informative requires d(forward refs, MDN) > 2x within-forward "
        "distance on the selected axis. Held types are reported, never "
        "auto-promoted; promotion stays a curator decision.")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"axis {axis_info['selected_axis']}, gap "
          f"{axis_info['gap_per_axis'][axis_info['selected_axis']]:.3f}")
    if out["separation"]:
        s = out["separation"]
        print(f"within-forward {s['within_forward_axis_dist']:.4f}, "
              f"forward->MDN {s['forward_to_mdn_axis_dist']:.4f}, "
              f"ratio {s['ratio']:.2f} -> leaf_informative={out['leaf_informative']}")
    print(f"{len(out['held_classification'])} held locomotion types classified; wrote {OUT}")


if __name__ == "__main__":
    main()
