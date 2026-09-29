# Kernel realism: data availability notes

## Gap junctions (parked — decision 2026-09-29)

- Kernel support exists: `Connectome.gap_edges` (diffusive CSR) +
  `params.gap_gain_fp` (0 = off). See `crates/brain/src/lib.rs`.
- **No source data**: MaleCNS v1.0 flat files (weights, syn-points,
  annotations, neurotransmitters — see `bridge/data/`) contain chemical
  synapses only. FlyWire v783 likewise ships no gap-junction table.
- Decision: stays parked. A gap source would need an external dataset
  (e.g. electrophysiology-inferred electrical coupling tables). Until
  one exists in `bridge/data/`, this stays off; no planned result
  depends on it.

## Axonal conduction delays

- Current model: per-edge delay from soma-position euclidean distance
  (`positions.f32`, per-body mean of pre-synapse positions), speed
  0.5 mm/ms, clamped [1, `max_delay_ticks`]. Uncovered positions (0.6%)
  clamp to the 1-tick minimum.
- Decision 2026-09-29: skeleton-based refinement (path length along the
  arbor instead of straight-line distance) stays parked — it needs a
  bulk skeleton download from the flyem bucket for MaleCNS body IDs,
  and no current result turns on straight-line vs arbor-path delay
  error. Revisit if a result shows delay-order sensitivity (e.g. a
  protocol where 2-3 tick jitter changes an arbiter outcome); the fetch
  is the first step then, before any kernel change.

## Behavioral macro-architecture note (2026-09-29)

The chemotaxis lock-on round (commit history: opponent weights ->
balance -> cast-surge) established that the instance-level steering
decode carries the chemotactic bearing signal at ~+4 Hz mean against
~100-300 Hz per-instance differential noise at the walk plant — an SNR
no continuous controller can use. The assay now runs the cast-surge
macro-controller (`WorldConfig::cast_surge`): concentration-anchored
casts (bilateral antenna difference, translation-invariant) + surges +
local search; the DN steering decode remains available with
`cast_surge: false` and still drives the optomotor/visual assays. A
future kernel whose DN population encodes bearing at usable SNR could
re-enable continuous steering; the measurement to beat is the
steer-drive distribution in `runs/calib-*` / `runs/lockon3-male`
traces.

