# Kernel realism: data availability notes

## Gap junctions (parked)

- Kernel support exists: `Connectome.gap_edges` (diffusive CSR) +
  `params.gap_gain_fp` (0 = off). See `crates/brain/src/lib.rs`.
- **No source data**: MaleCNS v1.0 flat files (weights, syn-points,
  annotations, neurotransmitters — see `bridge/data/`) contain chemical
  synapses only. FlyWire v783 likewise ships no gap-junction table.
- A gap source would need an external dataset (e.g. electrophysiology-
  inferred electrical coupling tables). Until one exists in
  `bridge/data/`, this stays parked.

## Axonal conduction delays

- Current model: per-edge delay from soma-position euclidean distance
  (`positions.f32`, per-body mean of pre-synapse positions), speed
  0.5 mm/ms, clamped [1, `max_delay_ticks`]. Uncovered positions (0.6%)
  clamp to the 1-tick minimum.
- Skeleton-based refinement (path length along the arbor instead of
  straight-line distance) needs a skeleton source for MaleCNS body IDs
  (neuroglancer precomputed skeletons). Not present in `bridge/data/`;
  would require a bulk download from the flyem bucket. Parked until that
  fetch is justified.
