#!/usr/bin/env python3
"""Extract a signed, weighted connectome from MaleCNS v1.0 flat files.

Inputs (gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/):
  - body-annotations-male-cns-v1.0-minconf-0.5.feather
  - body-neurotransmitters-male-cns-v1.0.feather
  - connectome-weights-male-cns-v1.0-minconf-0.5.feather

Output: connectome directory loadable from Rust:
  neurons.parquet   per-neuron metadata (type, superclass, nt, sign, flags)
  body_ids.u64      n × u64  MaleCNS bodyId (sorted; all arrays aligned)
  func_classes.u8   n × u8   0=inter, 1=sensory, 2=dn, 3=motor
  csr_offsets.u64   (n+1) × u64  row offsets into edge arrays
  csr_targets.u32   m × u32  target neuron index
  csr_weights.i16   m × i16  signed fixed-point weight (SCALE = 512)
  dn_types.json     descending-neuron type list (behavior readout seed)
  manifest.json     params, counts, input/output hashes

Neuron policy: keep every curated segment (status + superclass) as a node.
Transmitter: per-body prediction -> cell-type consensus -> motor-neuron
default (vnc_motor/cb_motor are glutamatergic; their predictions fail
because they synapse peripherally). Edges from still-unknown transmitters
are dropped; such neurons keep their incoming edges.

Weight model (v1):  w_fp = clamp(round(sign_mult * 512 * log2(1 + syn_count)))
  sign_mult: acetylcholine +1, gaba -1, glutamate -1, histamine -1 (insect
  photoreceptor histamine gates Cl-), modulators (dopamine/serotonin/
  octopamine/tyramine) +0.5 (assumption, tunable).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"

ANNOT = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NTFILE = "body-neurotransmitters-male-cns-v1.0.feather"
WFILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Statuses that identify a segment worth simulating (MaleCNS vocabulary).
KEEP_STATUS = {
    "Reviewed", "Roughly traced", "Prelim Roughly traced", "RT Hard to trace",
    "Hard to trace", "Partially traced", "0.5assign",
    "Anchor", "Soma Anchor", "Sensory Anchor",
}
SEN = {"ol_sensory", "cb_sensory", "vnc_sensory", "cb_sensory_tbc",
       "vnc_sensory_tbc", "sensory_ascending", "sensory_ascending_tbc",
       "sensory_descending"}
DN = {"descending_neuron", "descending_neuron_tbc", "efferent_descending"}
MN = {"vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "vnc_tbc"}
# Motor neurons are glutamatergic even when the predictor returns 'unclear'.
MOTOR_DEFAULT_NT = {"vnc_motor", "cb_motor"}

SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0,
        "dopamine": 0.5, "serotonin": 0.5, "octopamine": 0.5, "tyramine": 0.5}
SCALE = 512  # fixed-point scale for i16 weights


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def counts_dict(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def main() -> None:
    # --- neuron set: curated status + known superclass --------------------
    an = feather.read_table(DATA / ANNOT).to_pandas()
    nt = feather.read_table(
        DATA / NTFILE,
        columns=["body", "predicted_nt", "predicted_nt_confidence",
                 "celltype_predicted_nt"],
    ).to_pandas()

    neurons = an[an.statusLabel.isin(KEEP_STATUS) & an.superclass.notna()].copy()
    neurons = neurons.merge(nt, left_on="bodyId", right_on="body", how="left")
    neurons["nt"] = neurons.predicted_nt.where(
        neurons.predicted_nt != "unclear", neurons.celltype_predicted_nt)
    neurons.loc[neurons.nt == "unclear", "nt"] = None
    neurons["nt_conf"] = neurons.predicted_nt_confidence

    def fn(sc):
        return ("sensory" if sc in SEN else
                "dn" if sc in DN else
                "motor" if sc in MN else "inter")

    neurons["func_class"] = neurons.superclass.map(fn)
    motor_defaulted = neurons.nt.isna() & neurons.superclass.isin(MOTOR_DEFAULT_NT)
    neurons["eff_nt"] = neurons.nt
    neurons.loc[motor_defaulted, "eff_nt"] = "glutamate"
    print(f"neuron nodes (status+superclass): {len(neurons)}")
    print(f"  transmitter: per-body/cell-type known: {neurons.nt.notna().sum()}, "
          f"motor defaulted to glutamate: {motor_defaulted.sum()}, "
          f"unknown (no outgoing edges): {neurons.eff_nt.isna().sum()}")

    # --- per-position lookup tables over sorted body ids ------------------
    ids = np.sort(neurons.bodyId.to_numpy())
    n_ids = len(ids)
    recs = neurons.set_index("bodyId")
    sign_lut = np.array(
        [SIGN.get(t, np.nan) for t in recs.loc[ids].eff_nt])
    sup_lut = recs.loc[ids].superclass.to_numpy()
    fc_lut = recs.loc[ids].func_class.map(
        {"inter": 0, "sensory": 1, "dn": 2, "motor": 3}).to_numpy(dtype=np.uint8)
    sup_names = sorted(neurons.superclass.unique())
    sup_code = {s: i for i, s in enumerate(sup_names)}
    sup_codes = recs.loc[ids].superclass.map(sup_code).to_numpy(dtype=np.uint8)
    # somaSide, falling back to rootSide (ORNs and many sensory neurons have
    # no soma annotation but a segmented root side: L/R of their entry)
    ss = neurons.somaSide.where(
        neurons.somaSide.isin(["L", "R", "midline"]), neurons.rootSide)
    ss = ss.where(ss.isin(["L", "R", "midline"]), None)
    side_codes = (ss.fillna("none").map(
        {"none": 0, "L": 1, "R": 2, "midline": 3}).to_numpy(dtype=np.uint8))
    nerve_names = sorted(
        str(n) for n in neurons.entryNerve.fillna("none").unique())
    nerve_code = {n: i for i, n in enumerate(nerve_names)}
    nerve_codes = (neurons.entryNerve.fillna("none").map(nerve_code)
                   .to_numpy(dtype=np.uint16))
    type_names = sorted(str(t) for t in neurons.type.dropna().unique())
    type_code = {t: i for i, t in enumerate(type_names)}
    type_codes = (recs.loc[ids]["type"]
                  .map(lambda t: type_code.get(str(t), 65535))
                  .to_numpy(dtype=np.uint16))

    # --- filter weights: both ends in neuron set --------------------------
    w = feather.read_table(DATA / WFILE).to_pandas()
    total_raw_edges, total_raw_syn = len(w), int(w.weight.sum())
    mask = np.isin(w.body_pre.to_numpy(), ids) & np.isin(w.body_post.to_numpy(), ids)
    w = w[mask]
    autapses = int((w.body_pre == w.body_post).sum())
    w = w[w.body_pre != w.body_post]
    w = w.sort_values(["body_pre", "body_post"], kind="stable").reset_index(drop=True)

    src_pos = np.searchsorted(ids, w.body_pre.to_numpy())
    tgt_pos = np.searchsorted(ids, w.body_post.to_numpy())
    assert (ids[src_pos] == w.body_pre).all() and (ids[tgt_pos] == w.body_post).all()

    # drop edges whose presynaptic transmitter is unknown (sign undefined)
    sign_raw = sign_lut[src_pos]
    known_edge = ~np.isnan(sign_raw)
    dropped_unknown = int((~known_edge).sum())
    w, src_pos, tgt_pos = w[known_edge], src_pos[known_edge], tgt_pos[known_edge]
    sign_mult = sign_lut[src_pos]

    n_edges, n_syn = len(w), int(w.weight.sum())
    print(f"raw edges (all segments): {total_raw_edges:,} ({total_raw_syn:,} synapses)")
    print(f"edges kept: {n_edges:,} ({n_syn:,} synapses); "
          f"autapses dropped: {autapses}, unknown-nt edges dropped: {dropped_unknown}")

    # --- signed fixed-point weights ----------------------------------------
    w_fp = np.rint(sign_mult * SCALE * np.log2(1.0 + w.weight.to_numpy()))
    w_fp = np.clip(w_fp, -32768, 32767).astype(np.int16)
    syn_signed = sign_mult * w.weight.to_numpy()

    # --- CSR (rows = presynaptic, sorted by (pre, post)) ------------------
    counts = np.bincount(src_pos, minlength=n_ids)
    offsets = np.zeros(n_ids + 1, dtype=np.uint64)
    np.cumsum(counts, out=offsets[1:])
    tgt_idx = tgt_pos.astype(np.uint32)

    # --- E/I balance by presynaptic superclass ----------------------------
    sc_pre = pd.Series(sup_lut[src_pos])
    ei = pd.DataFrame({"sc": sc_pre, "syn": syn_signed}).groupby("sc").syn.agg(
        total="sum", excitatory=lambda s: s[s > 0].sum(),
        inhibitory=lambda s: -s[s < 0].sum())
    print("\nE/I (signed synapse sums) by presynaptic superclass:")
    print(ei.to_string(float_format=lambda x: f"{x:,.0f}"))

    # --- write outputs -----------------------------------------------------
    out_dir = OUT / "connectome-malecns-v1.0"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = ["bodyId", "type", "instance", "superclass", "somaSide",
                 "nt", "eff_nt", "nt_conf", "func_class", "statusLabel"]
    neurons[keep_cols].sort_values("bodyId").reset_index(drop=True).to_parquet(
        out_dir / "neurons.parquet", index=False)
    offsets.tofile(out_dir / "csr_offsets.u64")
    tgt_idx.tofile(out_dir / "csr_targets.u32")
    w_fp.tofile(out_dir / "csr_weights.i16")
    ids.tofile(out_dir / "body_ids.u64")
    fc_lut.tofile(out_dir / "func_classes.u8")
    sup_codes.tofile(out_dir / "superclass_codes.u8")
    (out_dir / "superclass_names.json").write_text(json.dumps(sup_names, indent=1))
    side_codes.tofile(out_dir / "side_codes.u8")
    nerve_codes.tofile(out_dir / "entry_nerve_codes.u16")
    (out_dir / "entry_nerve_names.json").write_text(
        json.dumps(nerve_names, indent=1))
    type_codes.tofile(out_dir / "type_codes.u16")
    (out_dir / "type_names.json").write_text(json.dumps(type_names, indent=1))

    dn_types = sorted(t for t in neurons[neurons.func_class == "dn"].type.dropna().unique())
    (out_dir / "dn_types.json").write_text(json.dumps(dn_types, indent=1))

    manifest = {
        "name": "connectome-malecns-v1.0",
        "inputs": {f: sha256(DATA / f) for f in (ANNOT, NTFILE, WFILE)},
        "params": {
            "keep_status": sorted(KEEP_STATUS),
            "weight_model": "clamp(round(sign * 512 * log2(1+count)))",
            "sign_map": SIGN,
            "unknown_nt_policy": "keep-node-drop-out-edges",
            "motor_default_nt": sorted(MOTOR_DEFAULT_NT),
            "autapses": "dropped",
        },
        "counts": {
            "neurons": n_ids, "edges": int(n_edges), "synapses": n_syn,
            "autapses_dropped": autapses,
            "unknown_nt_edges_dropped": dropped_unknown,
            "neurons_motor_defaulted": int(motor_defaulted.sum()),
            "neurons_unknown_nt": int(neurons.eff_nt.isna().sum()),
            "raw_edges_all_segments": total_raw_edges,
            "raw_synapses_all_segments": total_raw_syn,
            "per_func_class": counts_dict(neurons.func_class),
            "per_eff_nt": counts_dict(neurons.eff_nt),
            "per_superclass": counts_dict(neurons.superclass),
            "superclasses": sup_names,
            "dn_types": len(dn_types),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in ("neurons.parquet", "body_ids.u64", "csr_offsets.u64",
              "csr_targets.u32", "csr_weights.i16", "func_classes.u8",
              "superclass_codes.u8", "superclass_names.json",
              "side_codes.u8", "entry_nerve_codes.u16",
              "entry_nerve_names.json", "type_codes.u16", "type_names.json",
              "manifest.json"):
        manifest.setdefault("outputs", {})[f] = sha256(out_dir / f)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
