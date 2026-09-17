//! Engine-independent simulation core: brain + arbiter + body + arena.
//!
//! One `Sim::step` = 1 ms. The bevy renderer is a thin observer over this;
//! all behavior logic lives here so it is testable and batchable headless.

use std::sync::Arc;

use brain::{Brain, LifParams};
use connectome::Connectome;
use transduce::{
    apply_odor_gains, apply_visual_gains, BehaviorMapFile, DnArbiter, Drives, OdorField,
    Ommatidia, VisObject,
};

#[derive(Clone, Debug)]
pub struct WorldConfig {
    pub arena_radius_mm: f64,
    /// Open-loop mode: keep the kernel + arbiter running but do not move
    /// the body (visual/odor sampling at the fixed initial pose).
    #[doc(hidden)]
    pub freeze_body: bool,
    pub walk_speed_mm_s: f64,
    pub back_speed_mm_s: f64,
    /// Walk drive (Hz) at which walking reaches half speed.
    pub walk_half_hz: f64,
    /// rad/s at full steering drive.
    pub steer_gain_rad_s: f64,
    pub escape_speed_mm_s: f64,
    /// DNp01 EMA rate (Hz) that triggers escape. Burst detection: the
    /// giant fiber idles near 15 Hz under global (per-class) calibration,
    /// so the trigger sits above idle and must re-arm after firing.
    pub escape_trigger_hz: f64,
    pub escape_duration_ms: u64,
    /// Odor transducer: gain multiplier slope (rate multiplier per conc unit).
    pub odor_k: f64,
    /// Superclass code of the olfactory sensory class (cb_sensory).
    pub odor_superclass: u8,
    pub antenna_offset_mm: f64,
    /// Walk-speed multiplier per unit mean antenna concentration
    /// (surge: flies walk faster toward rising odor).
    pub surge_gain: f64,
    /// Visual gain slope (photoreceptor rate multiplier per luminance unit).
    pub visual_k: f64,
    /// Taste: sugar contact reach (mm) and gain slope for MxLbN afferents.
    pub sugar_reach_mm: f64,
    pub sugar_k: f64,
    /// Feed/groom state thresholds (EMA Hz over the respective class).
    pub feed_threshold_hz: f64,
    pub groom_threshold_hz: f64,
    /// Ommatidial acceptance angle (rad, Gaussian PSF sigma).
    pub visual_acceptance_rad: f64,
    pub seed: u64,
}

impl WorldConfig {
    pub fn for_con(con: &Connectome) -> Self {
        // male vocabulary: cb_sensory; FlyWire female: sensory
        let odor_superclass = con
            .superclass_names
            .iter()
            .position(|n| n == "cb_sensory" || n == "sensory")
            .expect("olfactory sensory superclass") as u8;
        WorldConfig {
            arena_radius_mm: 40.0,
            walk_speed_mm_s: 10.0,
            back_speed_mm_s: 6.0,
            walk_half_hz: 1.5,
            steer_gain_rad_s: 8.0,
            escape_speed_mm_s: 30.0,
            escape_trigger_hz: 25.0,
            escape_duration_ms: 250,
            odor_k: 12.0,
            odor_superclass,
            antenna_offset_mm: 0.6,
            surge_gain: 1.5,
            visual_k: 20.0,
            visual_acceptance_rad: 5.0 / 180.0 * std::f64::consts::PI,
            sugar_reach_mm: 2.0,
            sugar_k: 30.0,
            freeze_body: false,
            // calibrated MxLbN baseline is ~2.6 Hz mean: 3.0 left <1-sigma
            // margin and noise parked the fly in spurious "feed" states.
            // Real sugar contact boosts afferent gain ~31x (~80 Hz), so
            // 6.0 keeps a 7-sigma baseline margin with contact trivially
            // above it.
            feed_threshold_hz: 6.0,
            groom_threshold_hz: 5.0,
            seed: 7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Fly {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
}

/// Arbiter dynamics snapshot (wire-compatible with the historical
/// "arbiter" object in checkpoints).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ArbiterCp {
    pub ema: Vec<f64>,
    pub steer_dc: f64,
}

/// Typed full-embodiment snapshot. Wire-compatible with the historical
/// serde_json::Value layout (same keys, same nesting), so checkpoints
/// written by earlier versions still restore.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub tick: u64,
    pub brain: brain::BrainState,
    pub arbiter: ArbiterCp,
    pub fly: Fly,
    pub t_ms: u64,
    pub feed_ema: f64,
    pub groom_ema: f64,
    pub escape_until: u64,
    pub escape_armed: bool,
}

#[derive(Clone, Debug)]
pub struct StepInfo {
    pub t_ms: u64,
    pub fly: Fly,
    pub drives: Drives,
    pub v_mm_s: f64,
    pub w_rad_s: f64,
    pub label: &'static str,
}

pub struct Sim {
    brain: Brain,
    con: Arc<Connectome>,
    arbiter: DnArbiter,
    pub field: OdorField,
    pub vis_objects: Vec<VisObject>,
    pub ommatidia: Ommatidia,
    /// Optional sugar source (contact-activated taste).
    pub sugar: Option<(f64, f64)>,
    feed_ids: Vec<u32>,
    groom_ids: Vec<u32>,
    feed_ema: f64,
    groom_ema: f64,
    pub fly: Fly,
    cfg: WorldConfig,
    escape_until: u64,
    escape_armed: bool,
    t_ms: u64,
}

impl Sim {
    pub fn new(
        con: Arc<Connectome>,
        params: LifParams,
        map: &BehaviorMapFile,
        field: OdorField,
        cfg: WorldConfig,
    ) -> Self {
        let brain = Brain::new(Arc::clone(&con), params);
        let arbiter = DnArbiter::new(&con, map);
        let fly = Fly {
            x: 0.0,
            y: 0.0,
            heading: 0.0,
        };
        let mxbn = con
            .entry_nerve_names
            .iter()
            .position(|n| n == "MxLbN")
            .map(|code| code as u16)
            .unwrap_or(u16::MAX);
        let an = con
            .entry_nerve_names
            .iter()
            .position(|n| n == "AN")
            .map(|code| code as u16)
            .unwrap_or(u16::MAX);
        let mut feed_ids = Vec::new();
        let mut groom_ids = Vec::new();
        for i in 0..con.func_classes.len() as u32 {
            let j = i as usize;
            if con.func_classes[j] != connectome::FUNC_SENSORY {
                continue;
            }
            match con.entry_nerve_codes[j] {
                c if c == mxbn => feed_ids.push(i),
                c if c == an => groom_ids.push(i),
                _ => {}
            }
        }
        let mut ommatidia = Ommatidia::new(48);
        let ol_sensory = con
            .superclass_names
            .iter()
            .position(|n| n == "ol_sensory")
            .unwrap_or(usize::MAX) as u8;
        ommatidia.assign_neurons(&con, ol_sensory);
        Sim {
            brain,
            con,
            arbiter,
            field,
            vis_objects: Vec::new(),
            ommatidia,
            sugar: None,
            feed_ids,
            groom_ids,
            feed_ema: 0.0,
            groom_ema: 0.0,
            fly,
            cfg,
            escape_until: 0,
            escape_armed: true,
            t_ms: 0,
        }
    }

    pub fn drives(&self) -> Drives {
        self.arbiter.drives()
    }

    fn mxbn_code(&self) -> u16 {
        self.con
            .entry_nerve_names
            .iter()
            .position(|n| n == "MxLbN")
            .map(|c| c as u16)
            .unwrap_or(u16::MAX)
    }

    pub fn feed_groom_hz(&self) -> (f64, f64) {
        (self.feed_ema, self.groom_ema)
    }

    /// Install instance-level steering weights (from dn_screen output).
    pub fn apply_steer_instances(&mut self, insts: &[transduce::SteerInstance]) {
        self.arbiter.set_steer_instances(&self.con, insts);
    }

    /// Append a second modality's steering weights (e.g. visual pathway).
    pub fn add_steer_instances(&mut self, insts: &[transduce::SteerInstance]) {
        self.arbiter.add_steer_instances(&self.con, insts);
    }

    /// (bodyId, EMA rate Hz) per steering instance — centering support.
    pub fn steer_instance_stats(&self) -> Vec<(u64, f64)> {
        self.arbiter.steer_instance_stats(&self.con)
    }

    /// Measure each steering instance's no-stimulus rate in THIS arena
    /// (odor strength temporarily zeroed, same body/kernel state), then
    /// recenter bases so the static steering drive is nulled in this
    /// assay context. Advances the sim by warmup_ms + measure_ms ticks.
    /// Returns the pre-centering static bias (Hz-equivalent).
    pub fn center_steer_bases(&mut self, warmup_ms: u64, measure_ms: u64) -> f64 {
        use std::collections::HashMap;
        let saved = self.field;
        self.field = OdorField { x: 1.0e6, y: 1.0e6, strength: 0.0, radius_mm: 8.0 };
        for _ in 0..warmup_ms {
            self.step();
        }
        let mut sum: HashMap<u64, f64> = HashMap::new();
        let mut samples = 0u64;
        for t in 0..measure_ms {
            self.step();
            if t % 100 == 0 {
                for (body, hz) in self.steer_instance_stats() {
                    *sum.entry(body).or_default() += hz;
                }
                samples += 1;
            }
        }
        self.field = saved;
        let rate_of: HashMap<u64, f64> =
            sum.into_iter().map(|(b, s)| (b, s / samples as f64)).collect();
        self.arbiter.recenter_steer_bases(&self.con, &rate_of)
    }



    pub fn config(&self) -> &WorldConfig {
        &self.cfg
    }

    /// Snapshot the full embodied state (kernel + arbiter + body).
    pub fn checkpoint(&self) -> Checkpoint {
        let bs = self.brain.state();
        let (ema, sdc) = self.arbiter.state();
        Checkpoint {
            tick: bs.tick,
            brain: bs,
            arbiter: ArbiterCp { ema, steer_dc: sdc },
            fly: self.fly,
            t_ms: self.t_ms,
            feed_ema: self.feed_ema,
            groom_ema: self.groom_ema,
            escape_until: self.escape_until,
            escape_armed: self.escape_armed,
        }
    }

    /// Restore from [`Sim::checkpoint`]. Deterministic continuation.
    pub fn restore(&mut self, cp: &Checkpoint) {
        let mut bs = cp.brain.clone();
        bs.tick = cp.tick;
        self.brain.set_state(bs);
        self.arbiter.set_state(cp.arbiter.ema.clone(), cp.arbiter.steer_dc);
        self.fly = cp.fly;
        self.escape_until = cp.escape_until;
        self.escape_armed = cp.escape_armed;
        self.t_ms = cp.t_ms;
        self.feed_ema = cp.feed_ema;
        self.groom_ema = cp.groom_ema;
    }
    /// Advance 1 ms.
    pub fn step(&mut self) -> StepInfo {
        // 1. world -> sensory gains
        let (fx, fy, h) = (self.fly.x, self.fly.y, self.fly.heading);
        let gains = self.brain.sensory_gains_mut();
        apply_odor_gains(
            &self.con,
            gains,
            self.cfg.odor_superclass,
            &self.field,
            (fx, fy, h),
            self.cfg.antenna_offset_mm,
            self.cfg.odor_k,
        );

        // 1a. taste: sugar contact boosts MxLbN afferent gains
        if let Some((sx, sy)) = self.sugar {
            let near = ((fx - sx).powi(2) + (fy - sy).powi(2)).sqrt()
                <= self.cfg.sugar_reach_mm;
            let gains = self.brain.sensory_gains_mut();
            for &i in &self.feed_ids {
                gains[i as usize] = if near { 1.0 + self.cfg.sugar_k as f32 } else { 1.0 };
            }
        }

        // 1b. moving objects advance one tick before visual sampling
        //     (optomotor drum: VisObject::advance rotates about its pivot)
        for o in &mut self.vis_objects {
            o.advance(1);
        }

        // 1c. vision: ommatidial luminance -> photoreceptor gains
        let lum = self
            .ommatidia
            .luminance(&self.vis_objects, (fx, fy, h), self.cfg.visual_acceptance_rad);
        apply_visual_gains(
            &self.ommatidia,
            &lum,
            self.brain.sensory_gains_mut(),
            self.cfg.visual_k,
        );

        // 2. brain tick
        let spikes = self.brain.tick(&[]).to_vec();
        self.arbiter.observe(&spikes);
        // class-rate EMAs for feed/groom drives
        const FG_TAU: f64 = 500.0;
        let fg_alpha = 1.0 / (1.0 + FG_TAU);
        let (nf, ng) = (self.feed_ids.len() as f64, self.groom_ids.len() as f64);
        let mut sf = 0u32;
        let mut sg = 0u32;
        for &s in &spikes {
            if self.con.entry_nerve_codes[s as usize] == self.mxbn_code() {
                sf += 1;
            }
        }
        for &s in &spikes {
            if self.groom_ids.binary_search(&s).is_ok() {
                sg += 1;
            }
        }
        self.feed_ema += fg_alpha * ((sf as f64 / nf.max(1.0)) * 1000.0 - self.feed_ema);
        self.groom_ema += fg_alpha * ((sg as f64 / ng.max(1.0)) * 1000.0 - self.groom_ema);
        let d = self.arbiter.drives();

        // 3. behavior -> motion; surge from mean antenna concentration
        let cl = self.field.conc(fx - self.cfg.antenna_offset_mm * h.sin(),
                                 fy + self.cfg.antenna_offset_mm * h.cos());
        let cr = self.field.conc(fx + self.cfg.antenna_offset_mm * h.sin(),
                                 fy - self.cfg.antenna_offset_mm * h.cos());
        let mean_conc = 0.5 * (cl + cr);
        let (v, w, label) = self.motion(d, mean_conc);

        // 4. integrate + arena bound (skipped in open-loop mode)
        if self.cfg.freeze_body {
            self.t_ms += 1;
            return StepInfo {
                t_ms: self.t_ms,
                fly: self.fly,
                drives: d,
                v_mm_s: 0.0,
                w_rad_s: 0.0,
                label: "open-loop",
            };
        }
        self.fly.heading =
            (self.fly.heading + w / 1000.0).rem_euclid(2.0 * std::f64::consts::PI);
        let nx = self.fly.x + v / 1000.0 * self.fly.heading.cos();
        let ny = self.fly.y + v / 1000.0 * self.fly.heading.sin();
        let r = (nx * nx + ny * ny).sqrt();
        if r <= self.cfg.arena_radius_mm {
            self.fly.x = nx;
            self.fly.y = ny;
        } else {
            // clamp to wall and steer toward center next ticks
            let scale = self.cfg.arena_radius_mm / r;
            self.fly.x = nx * scale;
            self.fly.y = ny * scale;
            let to_center = self.fly.y.atan2(self.fly.x);
            self.fly.heading = turn_toward(self.fly.heading, to_center, 0.05);
        }

        self.t_ms += 1;
        StepInfo {
            t_ms: self.t_ms,
            fly: self.fly,
            drives: d,
            v_mm_s: v,
            w_rad_s: w,
            label,
        }
    }

    fn motion(&mut self, d: Drives, mean_conc: f64) -> (f64, f64, &'static str) {
        if self.t_ms < self.escape_until {
            return (self.cfg.escape_speed_mm_s, 0.0, "escape");
        }
        // re-arm only after the rate falls back below half-trigger
        if d.escape < 0.5 * self.cfg.escape_trigger_hz {
            self.escape_armed = true;
        }
        if self.escape_armed && d.escape > self.cfg.escape_trigger_hz {
            self.escape_until = self.t_ms + self.cfg.escape_duration_ms;
            self.escape_armed = false;
            return (self.cfg.escape_speed_mm_s, 0.0, "escape");
        }
        let walk_n = d.walk / (d.walk + self.cfg.walk_half_hz);
        let back_n = d.walk_back / (d.walk_back + self.cfg.walk_half_hz);
        let surge = 1.0 + self.cfg.surge_gain * mean_conc.min(1.0);
        let v = (self.cfg.walk_speed_mm_s * walk_n - self.cfg.back_speed_mm_s * back_n) * surge;
        let w = self.cfg.steer_gain_rad_s * (d.steer / 50.0).tanh();
        // Feeding is CONTACT-driven: without a sugar source in range the
        // MxLbN afferents carry no food signal, and their spontaneous
        // bursts must not park the fly in a phantom feed state.
        let (fx, fy) = (self.fly.x, self.fly.y);
        let sugar_contact = match self.sugar {
            Some((sx, sy)) => {
                ((fx - sx).powi(2) + (fy - sy).powi(2)).sqrt()
                    <= self.cfg.sugar_reach_mm
            }
            None => false,
        };
        if sugar_contact && self.feed_ema > self.cfg.feed_threshold_hz {
            return (0.0, 0.0, "feed");
        }
        if self.groom_ema > self.cfg.groom_threshold_hz && walk_n < 0.3 {
            return (0.0, 0.0, "groom");
        }
        let label = if back_n > 0.5 && back_n > walk_n {
            "walk_back"
        } else if walk_n > 0.3 {
            "walk"
        } else {
            "rest"
        };
        (v, w, label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc as StdArc;

    fn make_sim() -> Sim {
        let path = Path::new("../../bridge/out/connectome-malecns-v1.0");
        if !path.is_dir() {
            panic!("artifact required for world tests");
        }
        let con = Connectome::load(path).unwrap();
        let params: brain::LifParams = serde_json::from_slice(
            &std::fs::read("../../params/homeostasis_v1.json").unwrap(),
        )
        .unwrap_or_else(|_| brain::LifParams::default_for(&con));
        let map: transduce::BehaviorMapFile = serde_json::from_slice(
            &std::fs::read("../../scenarios/behavior_map.json").unwrap(),
        )
        .unwrap();
        let field = transduce::OdorField { x: 25.0, y: 0.0, strength: 1.0, radius_mm: 8.0 };
        let cfg = WorldConfig { seed: 7, ..WorldConfig::for_con(&con) };
        let mut sim = Sim::new(StdArc::new(con), params, &map, field, cfg);
        if let Ok(insts) =
            serde_json::from_slice::<Vec<transduce::SteerInstance>>(&std::fs::read(
                "../../scenarios/steer_instances.json",
            )
            .unwrap())
        {
            sim.apply_steer_instances(&insts);
        }
        sim
    }

    /// Backward compatibility: the historical serde_json::Value layout
    /// (tick at top level, brain without its tick key) must restore.
    #[test]
    fn restore_legacy_checkpoint_shape() {
        let mut sim = make_sim();
        for _ in 0..500 {
            sim.step();
        }
        let cp = sim.checkpoint();
        let legacy = serde_json::json!({
            "tick": cp.tick,
            "brain": {
                "cur": cp.brain.cur, "v": cp.brain.v, "adap": cp.brain.adap,
                "refrac": cp.brain.refrac, "spikes": cp.brain.spikes,
                "sensory_gain": cp.brain.sensory_gain,
                "buckets": cp.brain.buckets, "stsd": cp.brain.stsd,
                "rng": cp.brain.rng,
            },
            "arbiter": {"ema": cp.arbiter.ema, "steer_dc": cp.arbiter.steer_dc},
            "fly": {"x": cp.fly.x, "y": cp.fly.y, "heading": cp.fly.heading},
            "t_ms": cp.t_ms,
            "feed_ema": cp.feed_ema,
            "groom_ema": cp.groom_ema,
            "escape_until": cp.escape_until,
            "escape_armed": cp.escape_armed,
        });
        let legacy: Checkpoint = serde_json::from_value(legacy).expect("legacy shape");
        let mut sim2 = make_sim();
        sim2.restore(&legacy);
        for _ in 0..300 {
            let a = sim.step();
            let b = sim2.step();
            assert_eq!(a.fly, b.fly, "legacy restore diverged");
        }
    }

    /// Acceptance: checkpoint -> continue == uninterrupted run, bit-identical.
    #[test]
    fn resume_is_bit_identical() {
        let mut sim = make_sim();
        for _ in 0..1500 {
            sim.step();
        }
        let cp = sim.checkpoint();
        // branch A: keep going
        let mut a_tail = Vec::new();
        for _ in 0..1500 {
            let i = sim.step();
            a_tail.push((i.t_ms, i.fly.x, i.fly.y, i.fly.heading));
        }
        // branch C: restore the same checkpoint, continue identically
        let mut sim2 = make_sim();
        sim2.restore(&cp);
        let mut c_tail = Vec::new();
        for _ in 0..1500 {
            let i = sim2.step();
            c_tail.push((i.t_ms, i.fly.x, i.fly.y, i.fly.heading));
        }
        assert_eq!(a_tail, c_tail, "resume diverged from uninterrupted run");
    }
}

/// Rotate `from` toward `to` by at most `max_step` radians.
fn turn_toward(from: f64, to: f64, max_step: f64) -> f64 {
    let mut diff = (to - from + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    diff = diff.clamp(-max_step, max_step);
    from + diff
}
