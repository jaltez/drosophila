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
            feed_threshold_hz: 3.0,
            groom_threshold_hz: 5.0,
            seed: 7,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fly {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
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



    pub fn config(&self) -> &WorldConfig {
        &self.cfg
    }

    /// Snapshot the full embodied state (kernel + arbiter + body).
    pub fn checkpoint(&self) -> serde_json::Value {
        let bs = self.brain.state();
        let (ema, sdc) = self.arbiter.state();
        serde_json::json!({
            "tick": bs.tick,
            "brain": {
                "cur": bs.cur, "v": bs.v, "adap": bs.adap, "refrac": bs.refrac,
                "spikes": bs.spikes, "sensory_gain": bs.sensory_gain,
                "rng": bs.rng,
            },
            "arbiter": {"ema": ema, "steer_dc": sdc},
            "fly": {"x": self.fly.x, "y": self.fly.y, "heading": self.fly.heading},
            "t_ms": self.t_ms,
            "feed_ema": self.feed_ema,
            "groom_ema": self.groom_ema,
            "escape_until": self.escape_until,
            "escape_armed": self.escape_armed,
        })
    }

    /// Restore from [`Sim::checkpoint`]. Deterministic continuation.
    pub fn restore(&mut self, cp: &serde_json::Value) {
        let bs = brain::BrainState {
            cur: serde_json::from_value(cp["brain"]["cur"].clone()).unwrap(),
            v: serde_json::from_value(cp["brain"]["v"].clone()).unwrap(),
            adap: serde_json::from_value(cp["brain"]["adap"].clone()).unwrap(),
            refrac: serde_json::from_value(cp["brain"]["refrac"].clone()).unwrap(),
            spikes: serde_json::from_value(cp["brain"]["spikes"].clone()).unwrap(),
            sensory_gain: serde_json::from_value(cp["brain"]["sensory_gain"].clone()).unwrap(),
            rng: serde_json::from_value(cp["brain"]["rng"].clone()).unwrap(),
            tick: serde_json::from_value(cp["tick"].clone()).unwrap(),
        };
        self.brain.set_state(bs);
        let ema: Vec<f64> = serde_json::from_value(cp["arbiter"]["ema"].clone()).unwrap();
        let sdc: f64 = serde_json::from_value(cp["arbiter"]["steer_dc"].clone()).unwrap();
        self.arbiter.set_state(ema, sdc);
        self.fly = Fly {
            x: serde_json::from_value(cp["fly"]["x"].clone()).unwrap(),
            y: serde_json::from_value(cp["fly"]["y"].clone()).unwrap(),
            heading: serde_json::from_value(cp["fly"]["heading"].clone()).unwrap(),
        };
        self.escape_until = serde_json::from_value(cp["escape_until"].clone()).unwrap();
        self.escape_armed = serde_json::from_value(cp["escape_armed"].clone()).unwrap();
        self.t_ms = serde_json::from_value(cp["t_ms"].clone()).unwrap();
        self.feed_ema = serde_json::from_value(cp["feed_ema"].clone()).unwrap();
        self.groom_ema = serde_json::from_value(cp["groom_ema"].clone()).unwrap();
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

        // 1b. vision: ommatidial luminance -> photoreceptor gains
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

        // 4. integrate + arena bound
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
        if self.feed_ema > self.cfg.feed_threshold_hz {
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
