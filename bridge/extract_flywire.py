#!/usr/bin/env python3
"""Extract a signed, weighted connectome from the FlyWire female adult brain
(Fafb v783 proofread release), in the same binary format as MaleCNS
(see extract.py) so the Rust connectome loader can read either artifact.

Inputs:
  - https://zenodo.org/records/10676866/files/proofread_connections_783.feather
  - https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/
    supplemental_files/Supplemental_file1_neuron_annotations.tsv
  - https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy

Output: connectome directory loadable from Rust (identical file set/layout):
  neurons.parquet   per-neuron metadata (FlyWire-native column names)
  body_ids.u64      n x u64  FlyWire root_id (sorted; all arrays aligned)
  func_classes.u8   n x u8   0=inter, 1=sensory, 2=dn, 3=motor
  csr_offsets.u64   (n+1) x u64  row offsets into edge arrays
  csr_targets.u32   m x u32  target neuron index
  csr_weights.i16   m x i16  signed fixed-point weight (SCALE = 512)
  dn_types.json     descending-neuron type list (behavior readout seed)
  manifest.json     params, counts, input/output hashes

FlyWire-specific policy (differences from MaleCNS, mirrored in manifest):
  - Node set: annotation rows with `status` NaN (this TSV flags problem
    segments as outlier_seg/outlier_bio instead of curating positives —
    the FlyWire analog of MaleCNS KEEP_STATUS is "not flagged as outlier")
    AND non-empty super_class.
  - func_class from super_class (substring match, lowercased):
    sensory <- 'sensory'/'visual'/'optic' (optic-lobe neurons are inside
    the female brain volume; the MaleCNS 'ol' equivalent), dn <- 'descending',
    motor <- 'motor', else inter.
  - side: TSV `side` column left/right/center -> 1/2/3; 'na'/missing -> 0.
  - Transmitter: top_nt (lowercased); no motor default (top_nt covers brain
    motor neurons). Unknown -> keep node, drop outgoing edges.
  - Edges: connections feather aggregated to one row per (pre, post) pair
    by summing syn_count, then restricted to the node set; autapses dropped.

Vocabulary (status, super_class, side, sign) is shared with extract.py via
bridge/vocab.py; unmapped values are reported loudly and fail the run under
--strict / VOCAB_STRICT=1.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import vocab
from vocab import SCALE, SIGN

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"

CONN = "proofread_connections_783.feather"
ANNOT = "Supplemental_file1_neuron_annotations.tsv"
IDS = "proofread_root_ids_783.npy"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def counts_dict(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def main() -> None:
    # --- neuron set: not outlier-flagged + known superclass ---------------
    an = pd.read_csv(DATA / ANNOT, sep="\t", low_memory=False)
    status_counts = an["status"].value_counts(dropna=False)
    print("annotation status vocabulary:")
    print(status_counts.to_string())
    status_rep = vocab.VocabReport("flywire status (drop filter)")
    keep_status = vocab.flywire_status_classifier(status_rep)
    keep = (an["status"].astype(object).map(keep_status).astype(bool)
            & an["super_class"].notna())
    neurons = an[keep].copy()
    # NOTE: FlyWire root_ids are ~7.2e17 > 2^53; mixing int64 (feather) with
    # uint64 (node ids) would promote comparisons to float64 and corrupt
    # isin/searchsorted. Unify on uint64 before any set operation.
    neurons["root_id"] = neurons["root_id"].astype(np.uint64)
    # realign everything to root_id-sorted order (all arrays stay positional)
    neurons = (neurons.set_index("root_id")
               .loc[np.sort(neurons["root_id"].to_numpy())].reset_index())
    node_ids = neurons["root_id"].to_numpy()
    n_ids = len(node_ids)
    # cross-check against the official proofread root-id list
    proof_ids = np.load(DATA / IDS).astype(np.uint64)
    an_ids = an["root_id"].astype(np.uint64).to_numpy()
    proof_not_in_tsv = int((~np.isin(proof_ids, an_ids)).sum())
    nodes_not_proofread = int((~np.isin(node_ids, proof_ids)).sum())
    print(f"neuron nodes (status unflagged + super_class): {len(neurons)}")
    print(f"  dropped outlier-flagged segments: {len(an) - len(neurons)}")
    print(f"  proofread root-id list: {len(proof_ids):,}; "
          f"not in annotations TSV: {proof_not_in_tsv}; "
          f"nodes outside that list: {nodes_not_proofread}")

    # --- per-position lookup tables over sorted root ids ------------------
    nt = neurons["top_nt"].astype("string").str.lower()
    neurons["eff_nt"] = nt.to_numpy()
    neurons["nt"] = nt.to_numpy()

    func_rep = vocab.VocabReport("flywire super_class -> func_class")
    fc = neurons["super_class"].map(vocab.flywire_func_classifier(func_rep))
    neurons["func_class"] = fc
    fc_lut = fc.map({"inter": 0, "sensory": 1, "dn": 2, "motor": 3}).to_numpy(dtype=np.uint8)
    sup_lut = neurons["super_class"].to_numpy()
    sup_names = sorted(neurons["super_class"].astype(str).unique())
    sup_code = {s: i for i, s in enumerate(sup_names)}
    sup_codes = (neurons["super_class"].astype(str).map(sup_code)
                 .to_numpy(dtype=np.uint8))
    side_rep = vocab.VocabReport("flywire side -> side_code")
    side_codes = (neurons["side"].map(vocab.flywire_side_classifier(side_rep))
                  .to_numpy(dtype=np.uint8))
    type_names = sorted(str(t) for t in neurons["cell_type"].dropna().unique())
    type_code = {t: i for i, t in enumerate(type_names)}
    type_codes = (neurons["cell_type"]
                  .map(lambda t: type_code.get(str(t), 65535))
                  .to_numpy(dtype=np.uint16))
    sign_lut = np.array(
        [SIGN.get(t, np.nan) for t in neurons["eff_nt"].fillna("")])

    n_motor_unknown = int(neurons.loc[fc == "motor", "eff_nt"].isna().sum())
    n_unknown_nt = int(neurons["eff_nt"].isna().sum())
    print(f"  transmitter: known: {n_ids - n_unknown_nt}, unknown "
          f"(no outgoing edges): {n_unknown_nt} (motor among them: {n_motor_unknown})")

    # Loud vocabulary gate: unmapped raw values are echoed above; strict
    # mode (VOCAB_STRICT=1 / --strict) refuses to build an artifact from
    # unrecognized vocabulary.
    vocab.finish([status_rep, func_rep, side_rep])

    # --- aggregate connections feather to (pre, post) pairs ----------------
    conn = feather.read_table(
        DATA / CONN,
        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    total_raw_rows = conn.num_rows
    raw = conn.group_by(["pre_pt_root_id", "post_pt_root_id"]).aggregate(
        [("syn_count", "sum")]).to_pandas()
    raw.columns = ["pre", "post", "syn_count"]
    raw["pre"] = raw["pre"].astype(np.uint64)
    raw["post"] = raw["post"].astype(np.uint64)
    raw_edges, raw_syn = len(raw), int(raw["syn_count"].sum())
    print(f"\nconnections: {total_raw_rows:,} rows -> "
          f"{raw_edges:,} (pre,post) pairs ({raw_syn:,} synapses)")

    mask = (np.isin(raw["pre"].to_numpy(), node_ids)
            & np.isin(raw["post"].to_numpy(), node_ids))
    w = raw[mask]
    autapses = int((w["pre"] == w["post"]).sum())
    w = w[w["pre"] != w["post"]]
    w = w.sort_values(["pre", "post"], kind="stable").reset_index(drop=True)

    src_pos = np.searchsorted(node_ids, w["pre"].to_numpy())
    tgt_pos = np.searchsorted(node_ids, w["post"].to_numpy())
    assert (node_ids[src_pos] == w["pre"].to_numpy()).all()
    assert (node_ids[tgt_pos] == w["post"].to_numpy()).all()

    # drop edges whose presynaptic transmitter is unknown (sign undefined)
    sign_raw = sign_lut[src_pos]
    known_edge = ~np.isnan(sign_raw)
    dropped_unknown = int((~known_edge).sum())
    drop_by_fc = pd.Series(fc_lut[src_pos][~known_edge]).value_counts().sort_index()
    print("  unknown-NT dropped edges by presynaptic func_class "
          "(0=inter,1=sensory,2=dn,3=motor): "
          + ", ".join(f"{k}:{v:,}" for k, v in drop_by_fc.items()))
    w, src_pos, tgt_pos = w[known_edge], src_pos[known_edge], tgt_pos[known_edge]
    sign_mult = sign_lut[src_pos]

    n_edges, n_syn = len(w), int(w["syn_count"].sum())
    print(f"edges kept: {n_edges:,} ({n_syn:,} synapses); "
          f"autapses dropped: {autapses}, unknown-nt edges dropped: {dropped_unknown}")

    # --- signed fixed-point weights ----------------------------------------
    w_fp = np.rint(sign_mult * SCALE * np.log2(1.0 + w["syn_count"].to_numpy()))
    w_fp = np.clip(w_fp, -32768, 32767).astype(np.int16)
    syn_signed = sign_mult * w["syn_count"].to_numpy()

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
    out_dir = OUT / "connectome-flywire-v783"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = ["root_id", "cell_type", "super_class", "cell_class", "side",
                 "top_nt", "eff_nt", "top_nt_conf", "func_class", "status",
                 "flow", "nerve", "dimorphism"]
    out = neurons[keep_cols].copy()
    out.sort_values("root_id").reset_index(drop=True).to_parquet(
        out_dir / "neurons.parquet", index=False)
    offsets.tofile(out_dir / "csr_offsets.u64")
    tgt_idx.tofile(out_dir / "csr_targets.u32")
    w_fp.tofile(out_dir / "csr_weights.i16")
    node_ids.tofile(out_dir / "body_ids.u64")
    fc_lut.tofile(out_dir / "func_classes.u8")
    sup_codes.tofile(out_dir / "superclass_codes.u8")
    (out_dir / "superclass_names.json").write_text(json.dumps(sup_names, indent=1))
    side_codes.tofile(out_dir / "side_codes.u8")
    type_codes.tofile(out_dir / "type_codes.u16")
    (out_dir / "type_names.json").write_text(json.dumps(type_names, indent=1))

    dn_types = sorted(str(t) for t in
                      neurons.loc[fc.to_numpy() == "dn", "cell_type"].dropna().unique())
    (out_dir / "dn_types.json").write_text(json.dumps(dn_types, indent=1))

    manifest = {
        "name": "connectome-flywire-v783",
        "inputs": {f: sha256(DATA / f) for f in (ANNOT, CONN, IDS)},
        "params": {
            "dataset": "FlyWire female adult brain (Fafb) proofread release 783",
            "keep_status_rule": "status is NaN (this TSV flags problem segments "
                                "as outlier_seg/outlier_bio; no positive curation "
                                "vocabulary like MaleCNS KEEP_STATUS)",
            "func_class_rule": "explicit sets in bridge/vocab.py (sensory: "
                               "sensory/optic/visual_projection/visual_centrifugal/"
                               "sensory_ascending; dn: descending; motor: motor; "
                               "else inter; unmapped -> inter + loud report)",
            "side_rule": "TSV side: left->1 right->2 center->3(m midline); na/none->0",
            "nt_rule": "top_nt lowercased; no motor default; unknown -> keep node, "
                       "drop outgoing edges",
            "edge_aggregation": "connections feather grouped by (pre,post) summing "
                                "syn_count before endpoint restriction",
            "weight_model": "clamp(round(sign * 512 * log2(1+count)))",
            "sign_map": SIGN,
            "autapses": "dropped",
        },
        "counts": {
            "neurons": n_ids, "edges": int(n_edges), "synapses": n_syn,
            "autapses_dropped": autapses,
            "unknown_nt_edges_dropped": dropped_unknown,
            "neurons_unknown_nt": n_unknown_nt,
            "neurons_motor_unknown_nt": n_motor_unknown,
            "annotation_rows": len(an),
            "outlier_flagged_dropped": len(an) - len(neurons),
            "raw_connection_rows": total_raw_rows,
            "raw_edges_all_segments": raw_edges,
            "raw_synapses_all_segments": raw_syn,
            "proofread_root_ids": int(len(proof_ids)),
            "proofread_ids_not_in_annotations": proof_not_in_tsv,
            "nodes_not_in_proofread_list": nodes_not_proofread,
            "per_func_class": counts_dict(pd.Series(fc.to_numpy())),
            "per_eff_nt": counts_dict(neurons["eff_nt"]),
            "per_superclass": counts_dict(neurons["super_class"]),
            "superclasses": sup_names,
            "dn_types": len(dn_types),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in ("neurons.parquet", "body_ids.u64", "csr_offsets.u64",
              "csr_targets.u32", "csr_weights.i16", "func_classes.u8",
              "superclass_codes.u8", "superclass_names.json",
              "side_codes.u8", "type_codes.u16", "type_names.json",
              "manifest.json"):
        manifest.setdefault("outputs", {})[f] = sha256(out_dir / f)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
