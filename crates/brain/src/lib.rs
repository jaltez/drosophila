//! LIF simulation kernel over a loaded connectome.
//!
//! Contract: spikes in (sensory injection), spikes out, fixed 1 ms tick,
//! bit-deterministic given identical inputs and state. Implementation lands
//! in phase 2 of the project plan (wgpu compute + rayon fallback).
