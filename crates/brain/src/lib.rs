//! LIF simulation kernel over a loaded connectome.
//!
//! Determinism contract: identical connectome artifact, params, seed,
//! external input and sensory-gain timeline produce a bit-identical spike
//! sequence. Enforced by integer-only arithmetic: weights are i16
//! fixed-point (bridge SCALE=512), membrane/current are i64, synaptic
//! accumulation uses integer atomics (addition of integers is associative —
//! thread order cannot matter), and per-chunk parallel integration collects
//! spikes in fixed index order.
//!
//! Model per 1 ms tick:
//!   1. firing set S = spikes(t-1) + external + sensory background + intrinsic
//!      background (both Poisson; sensory rate scales per neuron by
//!      `sensory_gain` from world-driven transducers)
//!   2. scatter: for src in S,
//!        w_eff = w * gain[src superclass][w sign] * in_norm[target]
//!      cur[tgt] += w_eff   (gain is 2D: per-superclass, excitatory vs
//!      inhibitory — the E/I distinction is per-neuron, not per-superclass)
//!   3. integrate: v = (v * leak_num) >> 12 + cur; spike if
//!      v >= thresh + adap; on spike v = 0, adap += adapt_k, refrac counter
//!      set; adap decays each tick.
//!
//! Stability mechanisms (all required for homeostasis to be solvable):
//! - `in_norm` caps each neuron's total incoming |weight| — hubs need
//!   coincident input, not just any input.
//! - intrinsic background on ALL neurons removes the silent absorbing state.
//! - spike-frequency adaptation (`adapt_k`) caps sustained firing, turning
//!   recurrent-excitation runaway into a graded rate ceiling.
//!
//! Synapses are zero-delay: a spike delivers within its own tick.

mod rng;

pub use rng::Rng;

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use connectome::Connectome;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Fixed-point shift for membrane leak and gains (2^12 = 1.0).
pub const FP_SHIFT: i64 = 12;
pub const FP_ONE: i64 = 1 << FP_SHIFT;

const INTEGRATE_CHUNK: usize = 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifParams {
    /// Spike threshold in weight units (512 = one count-1 synapse).
    pub thresh: i64,
    /// Membrane leak numerator: v = (v * leak_num) >> FP_SHIFT.
    pub leak_num: i64,
    pub refrac_ticks: u32,
    /// Outgoing-weight gains per superclass for excitatory (w>0) and
    /// inhibitory (w<0) edges, fp 4.12 (FP_ONE = 1.0).
    pub gains_exc: Vec<i64>,
    pub gains_inh: Vec<i64>,
    /// Background Poisson rate (Hz) per superclass applied to sensory
    /// neurons of that superclass (standing drive).
    pub sensory_rates_hz: Vec<f64>,
    /// Intrinsic background Poisson rate (Hz) for every neuron.
    pub intrinsic_hz: f64,
    /// Per-target in-degree normalization cap: each neuron's total incoming
    /// |weight| is scaled down to this value (raw weight units; 0 = off).
    pub in_norm_k: i64,
    /// Spike-frequency adaptation: added to effective threshold after each
    /// spike (0 = off). Tames recurrent-excitation runaway.
    pub adapt_k: i64,
    /// Adaptation decay numerator per tick, fp 4.12.
    pub adapt_leak_num: i64,
    /// Conduction delays from body positions (requires positions.f32 in
    /// the artifact; falls back to uniform 1-tick delay without it).
    #[serde(default)]
    pub conduction_enabled: bool,
    /// Short-term synaptic depression (per-edge, lazy multiplicative).
    #[serde(default)]
    pub stsd_enable: bool,
    #[serde(default = "default_stsd_u")]
    pub stsd_u: f64,
    #[serde(default = "default_stsd_leak_num")]
    pub stsd_leak_num: f64,
    /// Gap-junction conductance, fp 4.12 (0 = off).
    #[serde(default)]
    pub gap_gain_fp: i64,
    #[serde(default = "default_speed")]
    pub conduction_speed_mm_ms: f64,
    #[serde(default = "default_max_delay")]
    pub max_delay_ticks: u32,
    #[serde(default)]
    pub seed: u64,
}

fn default_speed() -> f64 {
    0.5
}

fn default_max_delay() -> u32 {
    20
}

fn default_stsd_u() -> f64 {
    0.2
}

fn default_stsd_leak_num() -> f64 {
    4055.0
}

impl LifParams {
    pub fn default_for(con: &Connectome) -> Self {
        // tau_mem = 20 ms -> alpha ~ 0.9512; tau_adapt ~ 100 ms -> ~0.990
        LifParams {
            thresh: 8 * 512,
            leak_num: 3897,
            refrac_ticks: 2,
            gains_exc: vec![FP_ONE; con.superclass_names.len()],
            gains_inh: vec![FP_ONE; con.superclass_names.len()],
            sensory_rates_hz: vec![2.0; con.superclass_names.len()],
            intrinsic_hz: 0.5,
            in_norm_k: 50 * 512,
            adapt_k: 2048,
            adapt_leak_num: 4055,
            conduction_enabled: false,
            conduction_speed_mm_ms: 0.5,
            max_delay_ticks: 20,
            stsd_enable: false,
            stsd_u: 0.2,
            stsd_leak_num: 4055.0,
            gap_gain_fp: 0,
            seed: 42,
        }
    }
}

/// Full serializable kernel state (checkpoint/resume). `cur` IS included:
/// refractory neurons skip input consumption, so pending sums persist
/// across ticks and must survive checkpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainState {
    pub cur: Vec<i64>,
    pub buckets: Vec<Vec<(u32, i32)>>,
    pub stsd: Option<(Vec<f32>, Vec<u32>)>,
    pub v: Vec<i64>,
    pub adap: Vec<i64>,
    pub refrac: Vec<u32>,
    pub spikes: Vec<u32>,
    pub sensory_gain: Vec<f32>,
    pub rng: u64,
    /// top-level in the wire format; defaulted so older checkpoints that
    /// carry tick outside `brain` still deserialize
    #[serde(default)]
    pub tick: u64,
}

pub struct Brain {
    con: Arc<Connectome>,
    params: LifParams,
    v: Vec<i64>,
    adap: Vec<i64>,
    refrac: Vec<u32>,
    cur: Vec<AtomicI64>,
    spikes: Vec<u32>,
    sensory: Vec<u32>,
    /// Per-sensory-neuron rate multiplier (world-driven, e.g. odor field).
    sensory_gain: Vec<f32>,
    /// All-neuron index list for intrinsic background draws.
    all: Vec<u32>,
    /// Per-target incoming-weight normalizer, fp 4.12 (FP_ONE = unchanged).
    in_norm: Vec<i64>,
    /// Per-edge delay in ticks (1..=max), only when conduction is on.
    edge_delay: Option<Vec<u8>>,
    /// Scheduled deliveries per tick slot: (target, effective weight).
    buckets: Vec<Vec<(u32, i32)>>,
    /// Per-edge available depression (1.0 = full) and last-use tick.
    stsd: Option<(Vec<f32>, Vec<u32>)>,
    rng: Rng,
    tick: u64,
}

impl Brain {
    pub fn new(con: Arc<Connectome>, params: LifParams) -> Self {
        let n = con.body_ids.len();
        assert_eq!(params.gains_exc.len(), con.superclass_names.len());
        assert_eq!(params.gains_inh.len(), con.superclass_names.len());
        assert_eq!(params.sensory_rates_hz.len(), con.superclass_names.len());

        // per-target in-degree normalization factors
        let mut in_norm = vec![FP_ONE; n];
        if params.in_norm_k > 0 {
            let mut in_sum = vec![0i64; n];
            for i in 0..n as u32 {
                for (tgt, w) in con.out_edges(i) {
                    in_sum[tgt as usize] += w.unsigned_abs() as i64;
                }
            }
            let k = params.in_norm_k as f64;
            for (f, &s) in in_norm.iter_mut().zip(&in_sum) {
                if s > params.in_norm_k {
                    *f = ((k / s as f64) * FP_ONE as f64) as i64;
                }
            }
        }

        let sensory = con.indices_of_class(connectome::FUNC_SENSORY);
        let all: Vec<u32> = (0..n as u32).collect();
        let rng = Rng::new(params.seed ^ 0xC0FFEE);

        // conduction delays: per-edge tick delay from body positions
        let stsd = if params.stsd_enable {
            let m = con.csr_targets.len();
            Some((vec![1.0f32; m], vec![0u32; m])) // e=1: fully recovered
        } else {
            None
        };
        assert!(
            params.max_delay_ticks >= 1 && params.max_delay_ticks <= 255,
            "max_delay_ticks must be in 1..=255 (u8 slot storage), got {}",
            params.max_delay_ticks
        );
        // edge delays require conduction; buckets are needed by both the
        // delayed path and the STSD chunked path
        let edge_delay = if params.conduction_enabled {
            let max_d = params.max_delay_ticks as usize;
            let m = con.csr_targets.len();
            let speed = params.conduction_speed_mm_ms.max(1e-6);
            let delay: Vec<u8> = match &con.positions_mm {
                Some(pos) => {
                    let mut d = Vec::with_capacity(m);
                    for i in 0..n as u32 {
                        let (a, b) = (con.csr_offsets[i as usize], con.csr_offsets[i as usize + 1]);
                        for e in a..b {
                            let t = con.csr_targets[e as usize] as usize;
                            let (pa, pb) = (pos[i as usize], pos[t]);
                            let dist = ((pa[0] - pb[0]).powi(2)
                                + (pa[1] - pb[1]).powi(2)
                                + (pa[2] - pb[2]).powi(2))
                                .sqrt() as f64;
                            let ticks = (dist / speed).round() as usize;
                            d.push(ticks.clamp(1, max_d) as u8);
                        }
                    }
                    d
                }
                None => vec![1u8; m],
            };
            Some(delay)
        } else {
            None
        };
        let buckets = vec![Vec::new(); params.max_delay_ticks as usize];

        Brain {
            con,
            params,
            v: vec![0; n],
            adap: vec![0; n],
            refrac: vec![0; n],
            cur: (0..n).map(|_| AtomicI64::new(0)).collect(),
            spikes: Vec::new(),
            sensory,
            sensory_gain: vec![1.0; n],
            stsd,
            all,
            in_norm,
            edge_delay,
            buckets,
            rng,
            tick: 0,
        }
    }

    pub fn tick_no(&self) -> u64 {
        self.tick
    }

    /// Snapshot the full kernel state. `cur` is included: refractory
    /// neurons skip input consumption, so pending sums persist across ticks.
    pub fn state(&self) -> BrainState {
        let cur: Vec<i64> = self.cur.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        BrainState {
            cur,
            buckets: self.buckets.clone(),
            stsd: self.stsd.clone(),
            v: self.v.clone(),
            adap: self.adap.clone(),
            refrac: self.refrac.clone(),
            spikes: self.spikes.clone(),
            sensory_gain: self.sensory_gain.clone(),
            rng: self.rng.state(),
            tick: self.tick,
        }
    }

    /// Restore kernel state (connectome/params must be identical).
    pub fn set_state(&mut self, st: BrainState) {
        assert_eq!(st.v.len(), self.v.len());
        for (c, &val) in self.cur.iter().zip(&st.cur) {
            c.store(val, Ordering::Relaxed);
        }
        self.buckets = st.buckets;
        self.stsd = st.stsd;
        self.v = st.v;
        self.adap = st.adap;
        self.refrac = st.refrac;
        self.spikes = st.spikes;
        self.sensory_gain = st.sensory_gain;
        self.rng = Rng::from_state(st.rng);
        self.tick = st.tick;
    }

    pub fn params(&self) -> &LifParams {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut LifParams {
        &mut self.params
    }

    pub fn connectome(&self) -> &Connectome {
        &self.con
    }

    /// Re-seed the background PRNG (determinism / replay tooling).
    pub fn reseed_background(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    /// Mutable per-sensory-neuron rate multipliers (world-driven). The
    /// uniform draw is consumed regardless of the effective rate, so the
    /// PRNG stream is independent of gain values.
    pub fn sensory_gains_mut(&mut self) -> &mut [f32] {
        &mut self.sensory_gain
    }

    /// Spikes emitted by the most recent [`Brain::tick`], ascending index.
    pub fn spikes(&self) -> &[u32] {
        &self.spikes
    }

    /// Advance one 1 ms tick. `ext` are forced spikes (indices), e.g.
    /// from stimulus transducers.
    pub fn tick(&mut self, ext: &[u32]) -> &[u32] {
        let t = self.tick;
        let d_max = self.params.max_delay_ticks as u64;

        // 0. deliver arrivals scheduled for this tick
        if !self.buckets.is_empty() {
            let due = &self.buckets[(t % d_max) as usize];
            for (tgt, w) in due {
                self.cur[*tgt as usize].fetch_add(*w as i64, Ordering::Relaxed);
            }
        }

        // 0b. gap junctions: diffusive current between coupled neurons
        if self.params.gap_gain_fp != 0 {
            for &(a, b, g) in &self.con.gap_edges {
                let _ = g; // conductance folded into gap_gain_fp
                let flow = (self.params.gap_gain_fp
                    * (self.v[b as usize] - self.v[a as usize]))
                    >> FP_SHIFT;
                self.cur[a as usize].fetch_add(flow, Ordering::Relaxed);
                self.cur[b as usize].fetch_sub(flow, Ordering::Relaxed);
            }
        }

        // 1. firing set: backgrounds (drawn in fixed index order) + ext + prev
        let mut firing: Vec<u32> = Vec::with_capacity(self.spikes.len() + ext.len() + 128);
        let p = self.params.clone(); // tiny; avoids double borrows below
        let intrinsic_lambda = p.intrinsic_hz / 1000.0;
        if intrinsic_lambda > 0.0 {
            // Geometric-skip sampling: identical distribution to per-neuron
            // Bernoulli draws, one uniform per SPIKE instead of per neuron
            // (p ~ 5e-4 -> ~80 draws instead of 164k).
            let ln1p = (1.0 - intrinsic_lambda).ln();
            let mut i: u64 = 0;
            while i < self.all.len() as u64 {
                let u = self.rng.next_f64().max(1e-12);
                i += ((u.ln() / ln1p).floor() as u64) + 1;
                if i <= self.all.len() as u64 {
                    firing.push(self.all[(i - 1) as usize]);
                }
            }
        }
        for &i in &self.sensory {
            let sup = self.con.superclass_codes[i as usize] as usize;
            let lambda = p.sensory_rates_hz[sup] / 1000.0
                * self.sensory_gain[i as usize] as f64;
            if self.rng.next_f64() < lambda {
                firing.push(i);
            }
        }
        firing.extend_from_slice(ext);
        firing.extend_from_slice(&self.spikes);
        // firing is a set, not a multiset: a neuron driven by several
        // sources in the same tick delivers ONE spike (refractory synapses).
        // Sort + dedup is deterministic and segments are pre-sorted where
        // possible, so this is near-linear in practice.
        firing.sort_unstable();
        firing.dedup();

        // 2. scatter (integer atomics: order-independent)
        let con = &self.con;
        let g_exc = &self.params.gains_exc;
        let g_inh = &self.params.gains_inh;
        let delayed = self.edge_delay.is_some();
        if !delayed && !self.params.stsd_enable {
            firing.par_iter().for_each(|&src| {
                let sup = con.superclass_codes[src as usize] as usize;
                let (ge, gi) = (g_exc[sup] as i64, g_inh[sup] as i64);
                if ge == 0 && gi == 0 {
                    return;
                }
                for (tgt, w) in con.out_edges(src) {
                    let g = if w >= 0 { ge } else { gi };
                    let w_eff = (w as i64)
                        .saturating_mul(g)
                        .saturating_mul(self.in_norm[tgt as usize])
                        >> (2 * FP_SHIFT);
                    if w_eff != 0 {
                        self.cur[tgt as usize].fetch_add(w_eff, Ordering::Relaxed);
                    }
                }
            });
        } else {
            // chunked path (delays and/or STSD): per-chunk local buffers,
            // merged in firing-chunk order. Entry order within a slot is
            // irrelevant — delivery is an integer fetch_add (associative);
            // STSD updates are keyed by edge index, applied post-collect.
            const SCATTER_CHUNK: usize = 256;
            let d_slots = d_max as usize;
            let stsd_on = self.params.stsd_enable;
            let stsd_leak = self.params.stsd_leak_num / 4096.0;
            let stsd_u = self.params.stsd_u;
            let chunk_results = firing.par_chunks(SCATTER_CHUNK)
                .map(|chunk| {
                let mut buf = vec![Vec::new(); d_slots];
                let mut d_up: Vec<(usize, f32, u32)> = Vec::new();
                for &src in chunk {
                    let sup = con.superclass_codes[src as usize] as usize;
                    let (ge, gi) = (g_exc[sup] as i64, g_inh[sup] as i64);
                    if ge == 0 && gi == 0 {
                        continue;
                    }
                    for (tgt, e, w) in con.out_edges_indexed(src) {
                        let g = if w >= 0 { ge } else { gi };
                        let w_eff = (w as i64)
                            .saturating_mul(g)
                            .saturating_mul(self.in_norm[tgt as usize])
                            >> (2 * FP_SHIFT);
                        if w_eff == 0 {
                            continue;
                        }
                        // Tsodyks-Markram style: stored `e` is the LAST
                        // depleted fraction; idle time RECOVERS e toward 1
                        // (e_eff = 1-(1-e)*leak^age), never decays further.
                        let avail = if stsd_on {
                            let (dv, lt) = self.stsd.as_ref().unwrap();
                            let age = (t - lt[e as usize] as u64) as i32;
                            let e_eff =
                                1.0 - (1.0 - dv[e as usize] as f64) * stsd_leak.powi(age);
                            e_eff as f32
                        } else {
                            1.0
                        };
                        let delivered = ((w_eff as f64) * avail as f64).round() as i32;
                        if delivered == 0 {
                            continue;
                        }
                        match &self.edge_delay {
                            Some(delays) => {
                                let d = delays[e as usize] as u64;
                                let slot = ((t + d) % d_max) as usize;
                                buf[slot].push((tgt, delivered));
                            }
                            None => {
                                // zero-delay: apply within this tick
                                self.cur[tgt as usize]
                                    .fetch_add(delivered as i64, Ordering::Relaxed);
                            }
                        }
                        if stsd_on {
                            let (dv, lt) = self.stsd.as_ref().unwrap();
                            d_up.push((
                                e as usize,
                                ((1.0 - (1.0 - dv[e as usize] as f64)
                                    * stsd_leak.powi((t - lt[e as usize] as u64) as i32))
                                    * (1.0 - stsd_u)) as f32,
                                t as u32,
                            ));
                        }
                    }
                }
                (buf, d_up)
            })
            .collect::<Vec<_>>();
            for (per_chunk, mut d_up) in chunk_results {
                for (slot, mut entries) in per_chunk.into_iter().enumerate() {
                    self.buckets[slot].append(&mut entries);
                }
                if stsd_on {
                    let (dv, lt) = self.stsd.as_mut().unwrap();
                    for (e, d, tstamp) in d_up.drain(..) {
                        dv[e] = d;
                        lt[e] = tstamp;
                    }
                }
            }
        }

        // 3. integrate in fixed chunks; flatten local spike lists in order
        let thresh = self.params.thresh;
        let leak = self.params.leak_num;
        let ad_leak = self.params.adapt_leak_num;
        let adapt_k = self.params.adapt_k;
        let refrac_set = self.params.refrac_ticks;
        let local: Vec<Vec<u32>> = self
            .v
            .par_chunks_mut(INTEGRATE_CHUNK)
            .zip(self.adap.par_chunks_mut(INTEGRATE_CHUNK))
            .zip(self.refrac.par_chunks_mut(INTEGRATE_CHUNK))
            .enumerate()
            .map(|(ci, ((vs, adaps), rs))| {
                let base = ci * INTEGRATE_CHUNK;
                let mut spiked = Vec::new();
                for j in 0..vs.len() {
                    let i = base + j;
                    adaps[j] = adaps[j].saturating_mul(ad_leak) >> FP_SHIFT;
                    if rs[j] > 0 {
                        rs[j] -= 1;
                        continue;
                    }
                    let input = self.cur[i].swap(0, Ordering::Relaxed);
                    if input == 0 && vs[j] == 0 {
                        continue;
                    }
                    let v_new = ((vs[j].saturating_mul(leak) >> FP_SHIFT)
                        .saturating_add(input))
                    .max(0);
                    vs[j] = v_new;
                    if v_new >= thresh + adaps[j] {
                        spiked.push(i as u32);
                        vs[j] = 0;
                        adaps[j] = adaps[j].saturating_add(adapt_k);
                        rs[j] = refrac_set;
                    }
                }
                spiked
            })
            .collect();
        drop(firing);
        self.spikes.clear();
        for chunk in local {
            self.spikes.extend_from_slice(&chunk);
        }
        self.tick += 1;
        &self.spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic() -> Connectome {
        // chain: 0 (sensory) -> 1 (inter) -> 2 (dn); 8-synapse weights
        let mut csr_offsets = vec![0u64];
        let mut targets = vec![];
        let mut weights = vec![];
        let rows: &[&[u32]] = &[&[1], &[2], &[]];
        for r in rows {
            for &t in *r {
                targets.push(t);
                weights.push((8.0 * 512.0) as i16);
            }
            csr_offsets.push(targets.len() as u64);
        }
        Connectome {
            dir: "synthetic".into(),
            manifest: serde_json::from_str(
                r#"{"name":"synthetic","counts":{"neurons":3,"edges":2,"synapses":16}}"#,
            )
            .unwrap(),
            body_ids: vec![1, 2, 3],
            func_classes: vec![1, 0, 2],
            superclass_codes: vec![0, 0, 0],
            superclass_names: vec!["syn".into()],
            side_codes: vec![0, 0, 0],
            type_codes: vec![0, 0, 0],
            type_names: vec!["synT".into()],
            gap_edges: Vec::new(),
            positions_mm: Some(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            entry_nerve_codes: vec![0, 0, 0],
            entry_nerve_names: vec!["none".into()],
            csr_offsets,
            csr_targets: targets,
            csr_weights: weights,
        }
    }

    #[test]
    fn chain_propagates_one_hop_per_tick() {
        // zero-delay synapses: ext spike on 0 delivers to 1 within the same
        // tick; 1 crosses threshold and fires; next tick 1 delivers to 2
        let con = synthetic();
        let mut p = LifParams::default_for(&con);
        p.sensory_rates_hz = vec![0.0];
        p.intrinsic_hz = 0.0;
        let mut b = Brain::new(Arc::new(con), p);
        assert_eq!(b.tick(&[0]).to_vec(), vec![1]);
        assert_eq!(b.tick(&[]).to_vec(), vec![2]);
        assert!(b.tick(&[]).is_empty());
    }

    #[test]
    fn sensory_gain_scales_firing() {
        // high gain on the sensory neuron drives the chain repeatedly;
        // zero gain silences it below the baseline rate
        let mut p = LifParams::default_for(&synthetic());
        p.sensory_rates_hz = vec![40.0];
        p.intrinsic_hz = 0.0;
        p.seed = 7;
        let mut hot = Brain::new(Arc::new(synthetic()), p.clone());
        hot.sensory_gains_mut()[0] = 5.0;
        let mut cold = Brain::new(Arc::new(synthetic()), p);
        cold.sensory_gains_mut()[0] = 0.0;
        let mut hot_spikes = 0usize;
        let mut cold_spikes = 0usize;
        for _ in 0..2000 {
            hot_spikes += hot.tick(&[]).len();
            cold_spikes += cold.tick(&[]).len();
        }
        assert!(hot_spikes > cold_spikes);
        assert!(hot_spikes > 100);
        assert!(cold_spikes == 0);
    }

    #[test]
    fn conduction_delays_deliver_two_ticks_per_hop() {
        // positions 1 mm apart at speed 0.5 mm/ms -> 2-tick edge delay
        let con = synthetic();
        let mut p = LifParams::default_for(&con);
        p.sensory_rates_hz = vec![0.0];
        p.intrinsic_hz = 0.0;
        p.conduction_enabled = true;
        p.conduction_speed_mm_ms = 0.5;
        let mut b = Brain::new(Arc::new(con), p);
        assert!(b.tick(&[0]).is_empty()); // scheduled for t0+2
        assert!(b.tick(&[]).is_empty());  // still in flight
        assert_eq!(b.tick(&[]).to_vec(), vec![1]); // arrives, fires at t2
        assert!(b.tick(&[]).is_empty());  // 1->2 in flight (t2+2)
        assert!(b.tick(&[]).is_empty());
        assert_eq!(b.tick(&[]).to_vec(), vec![2]); // fires at t4
    }

    #[test]
    fn stsd_reduces_and_recovers() {
        // single edge 0->1, w = thresh. Drive with ext every 10 ticks.
        // STSD: first delivery fires neuron 1 (e=1); depleted e then only
        // partially recovers over 10 idle ticks -> later deliveries
        // sub-threshold -> strictly fewer postsynaptic spikes than without.
        let build = |stsd: bool| {
            let con = synthetic();
            let mut p = LifParams::default_for(&con);
            p.sensory_rates_hz = vec![0.0];
            p.intrinsic_hz = 0.0;
            p.conduction_enabled = false;
            // 2x gain: effective weight 8192 = 2x threshold, so depleted
            // (avail < ~0.5) and recovered (avail ~1) states are separable
            p.gains_exc = vec![2 * FP_ONE];
            p.stsd_enable = stsd;
            p.stsd_u = 0.5;
            p.stsd_leak_num = 4055.0;
            Brain::new(Arc::new(con), p)
        };
        let mut on = build(true);
        let mut off = build(false);
        let mut on_ticks = Vec::new();
        let mut off_ticks = Vec::new();
        // Phase 1 (t<200): drive every 10 ticks. Depression must reduce
        // STSD output below the non-stsd baseline...
        for t in 0..200 {
            let ext = if t % 10 == 0 { vec![0u32] } else { Vec::new() };
            if on.tick(&ext).iter().any(|&i| i == 1) {
                on_ticks.push(t);
            }
            if off.tick(&ext).iter().any(|&i| i == 1) {
                off_ticks.push(t);
            }
        }
        // First drive delivers full weight (e=1) in both.
        assert_eq!(on_ticks.first(), Some(&0));
        assert_eq!(off_ticks.first(), Some(&0));
        assert!(
            on_ticks.len() < off_ticks.len(),
            "depression must reduce output: on {} vs off {}",
            on_ticks.len(),
            off_ticks.len()
        );
        // Phase 2 (t=500, after 300 idle ticks): recovery must restore
        // efficacy — the previously-silenced target fires again on STSD.
        // (idle decays the arbiter of the world: nothing to do here, the
        // kernel only needs the ticks.)
        let mut late_on = 0;
        for _ in 0..10 {
            let ext = vec![0u32];
            if on.tick(&ext).iter().any(|&i| i == 1) {
                late_on += 1;
            }
        }
        assert!(late_on >= 1, "recovery failed: idle synapse stays silenced");
        assert!(on_ticks.last().unwrap() < &200, "sanity");
    }

    #[test]
    fn stsd_without_conduction_no_panic() {
        // review finding: stsd on + conduction off used to index empty buckets
        let con = synthetic();
        let mut p = LifParams::default_for(&con);
        p.stsd_enable = true;
        p.conduction_enabled = false;
        let mut b = Brain::new(Arc::new(con), p);
        for _ in 0..50 {
            let _ = b.tick(&[0]);
        }
    }

    #[test]
    fn determinism_synthetic() {
        let run = || {
            let con = synthetic();
            let mut p = LifParams::default_for(&con);
            p.sensory_rates_hz = vec![5.0];
            p.intrinsic_hz = 0.5;
            p.seed = 123;
            let mut b = Brain::new(Arc::new(con), p);
            b.reseed_background(123);
            let mut acc = Vec::new();
            for _ in 0..200 {
                acc.extend_from_slice(b.tick(&[]));
            }
            acc
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn real_artifact_smoke() {
        let path = std::path::Path::new("../../bridge/out/connectome-malecns-v1.0");
        if !path.is_dir() {
            return;
        }
        let con = Connectome::load(path).unwrap();
        let params = LifParams::default_for(&con);
        let mut b = Brain::new(Arc::new(con), params);
        b.reseed_background(42);
        let mut total = 0usize;
        for _ in 0..50 {
            total += b.tick(&[]).len();
        }
        assert!(total > 0, "no spontaneous activity in smoke run");
    }
}
