//! Engine-independent simulation core: brain + arbiter + body + arena.
//!
//! One `Sim::step` = 1 ms. The bevy renderer is a thin observer over this;
//! all behavior logic lives here so it is testable and batchable headless.

use std::sync::Arc;

use brain::{Brain, LifParams};
use connectome::Connectome;
use transduce::{apply_odor_gains, BehaviorMapFile, DnArbiter, Drives, OdorField};

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
    pub seed: u64,
}

impl WorldConfig {
    pub fn for_con(con: &Connectome) -> Self {
        let odor_superclass = con
            .superclass_names
            .iter()
            .position(|n| n == "cb_sensory")
            .expect("cb_sensory superclass") as u8;
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
        Sim {
            brain,
            con,
            arbiter,
            field,
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

    /// Install instance-level steering weights (from dn_screen output).
    pub fn apply_steer_instances(&mut self, insts: &[transduce::SteerInstance]) {
        self.arbiter.set_steer_instances(&self.con, insts);
    }



    pub fn config(&self) -> &WorldConfig {
        &self.cfg
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

        // 2. brain tick
        let spikes = self.brain.tick(&[]).to_vec();
        self.arbiter.observe(&spikes);
        let d = self.arbiter.drives();

        // 3. behavior -> motion
        let (v, w, label) = self.motion(d);

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

    fn motion(&mut self, d: Drives) -> (f64, f64, &'static str) {
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
        let v = self.cfg.walk_speed_mm_s * walk_n - self.cfg.back_speed_mm_s * back_n;
        let w = self.cfg.steer_gain_rad_s * (d.steer / 50.0).tanh();
        let label = if back_n > 0.5 && back_n > walk_n {
            "walk_back"
        } else if walk_n > 0.3 {
            "walk"
        } else if d.feed > 5.0 {
            "feed"
        } else if d.groom > 5.0 {
            "groom"
        } else {
            "rest"
        };
        (v, w, label)
    }
}

/// Rotate `from` toward `to` by at most `max_step` radians.
fn turn_toward(from: f64, to: f64, max_step: f64) -> f64 {
    let mut diff = (to - from + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    diff = diff.clamp(-max_step, max_step);
    from + diff
}
