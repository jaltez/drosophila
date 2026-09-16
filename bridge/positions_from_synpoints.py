#!/usr/bin/env python3
"""Extract per-body mean pre-synapse positions (mm) from the MaleCNS
syn-points feather (streamed — file is 12.7 GB) and write positions.f32
(n×3, x,y,z in mm, aligned to body_ids.u64 order) into the artifact.
Also writes positions_present.json {covered: <n covered>}."""
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

D = Path(__file__).parent / "out" / "connectome-malecns-v1.0"
SRC = Path(__file__).parent / "data" / (
    "syn-points-male-cns-v1.0-minconf-0.5.feather")

body_ids = np.fromfile(D / "body_ids.u64", dtype=np.uint64)
n = len(body_ids)
idx = {int(b): i for i, b in enumerate(body_ids)}

sums = np.zeros((n, 3), dtype=np.float64)
counts = np.zeros(n, dtype=np.int64)

try:
    pf = feather.FeatherReader(str(SRC))
except Exception:
    pf = __import__("pyarrow").ipc.RecordBatchFileReader(SRC)
sch = pf.schema
names = sch.names
print("columns:", names)
body_col = "body" if "body" in names else "bodyId"
kind_col = "kind" if "kind" in names else None
pos_cols = [c for c in ("x", "y", "z") if c in names]

for bi in range(pf.num_record_batches):
    batch = pf.get_batch(bi)
    df = batch.to_pandas()
    if kind_col and kind_col in names:
        df = df[df[kind_col] == "PreSyn"]
    if df.empty:
        continue
    bodies = df[body_col].to_numpy()
    pts = df[pos_cols].to_numpy(dtype=np.float64)  # voxel units (8 nm)
    keep = np.isin(bodies, body_ids)
    if not keep.any():
        continue
    bodies, pts = bodies[keep], pts[keep]
    pos_idx = np.searchsorted(body_ids, bodies)
    np.add.at(sums, pos_idx, pts)
    np.add.at(counts, pos_idx, 1)

covered = int((counts > 0).sum())
mean_vox = np.divide(sums, counts[:, None],
                     out=np.zeros_like(sums), where=counts[:, None] > 0)
# voxel (8 nm) -> mm: x8e-3 nm /1000
pos_mm = (mean_vox * 8e-3 / 1000.0).astype(np.float32)
pos_mm.tofile(D / "positions.f32")

# Uncovered bodies are written as (0,0,0): with conduction delays enabled
# their efferent delay clamps to the 1-tick minimum ("unknown -> shortest").
meta = {"covered": covered, "total": n, "units": "mm"}
(D / "positions_present.json").write_text(json.dumps(meta, indent=1))
print(f"covered {covered}/{n} bodies with positions; wrote positions.f32")
