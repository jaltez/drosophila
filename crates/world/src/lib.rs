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
    /// Escape is a BURST response: trigger when the DNp01 drive exceeds a
    /// slow idle EMA (tau ~4 s) by this margin. An absolute threshold
    /// cannot work — under global (per-class) calibration the giant fiber
    /// idles within ~1 sigma of any fixed trigger, and Poisson noise
    /// crossed it every ~7 s (4.5 s of spurious escape per 30 s chemotaxis
    /// run, both odor and control arms). Must re-arm after firing.
    pub escape_burst_hz: f64,
    pub escape_duration_ms: u64,
    /// Odor transducer: gain multiplier slope (rate multiplier per conc unit).
    pub odor_k: f64,
    /// Scale (Hz) of the steering tanh: w = steer_gain * tanh(steer/this).
    /// The instance-sum steering drive is the weighted sum over ~10^3 DN
    /// instances; its noise sigma was measured at ~116 Hz in the 30 s
    /// chemotaxis assay, so the historical half-saturation at 50 Hz put
    /// 67% of walk ticks in bang-bang saturation (turn-rate limit cycles,
    /// fly pivoting in place). Default matches the measured drive scale.
    pub steer_scale_hz: f64,
    /// Cast-and-surge / local search. The instance-level steering decode
    /// carries the chemotactic signal at ~+4 Hz mean against ~100-300 Hz
    /// of per-instance differential noise (measured in the 30 s assay):
    /// at the walk plant (5-8 mm/s, turn-rate noise +-5 rad/s) the fly
    /// scribbles in place and no tanh rescale recovers a 30x SNR deficit.
    /// The fly algorithm for this regime is active sampling: on faint odor
    /// or losing a concentration peak, RUN A CAST — a programmed sinusoidal
    /// heading sweep (sweep_amp_rad_s at sweep_period_ms) during which the
    /// concentration-anchored surge bearing is recorded (the heading at
    /// max concentration); then SURGE straight toward that bearing for
    /// surge_ms. Outside cast/surge the DN steering decode runs unchanged.
    pub cast_surge: bool,
    /// Amplitude of the cast heading sweep (rad/s sinusoid). The
    /// antenna-optimal bearing sits ~90 deg away from the current heading
    /// when the fly faces the source, so the sweep excursion
    /// amp*period/2pi must exceed ~1.6 rad or the anchor degenerates to a
    /// fixed 53-deg turn (measured: excursion 0.64 -> constant 0.93 rad
    /// surge error at every distance/heading; >= 1.6 -> ~0.001 rad).
    pub sweep_amp_rad_s: f64,
    /// Period of one full cast sweep (ms); a cast lasts one period —
    /// a full sine swing already scans the antenna over the full
    /// +-excursion cone.
    pub sweep_period_ms: u64,
    /// Straight-run duration after a cast (ms).
    pub surge_ms: u64,
    /// Walk-speed multiplier during a cast (flies slow down to sample).
    pub cast_speed: f64,
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
    /// Grooming additionally requires near-zero odor context (the groom
    /// channel is the antennal nerve, whose afferents are the same
    /// olfactory population the odor drive boosts — without the gate the
    /// odor assay parked the fly in phantom grooming; contact-gated
    /// feeding above is unaffected since it keys on MxLbN).
    pub feed_threshold_hz: f64,
    pub groom_threshold_hz: f64,
    /// Ommatidial acceptance angle (rad, Gaussian PSF sigma).
    pub visual_acceptance_rad: f64,
    pub seed: u64,
}

impl WorldConfig {
    /// Struct defaults independent of any connectome (tests, tools).
    /// The olfactory superclass code is context: `for_con` resolves it
    /// from the loaded dataset vocabulary (male cb_sensory / FlyWire sensory).
    pub fn defaults() -> Self {
        WorldConfig {
            arena_radius_mm: 40.0,
            walk_speed_mm_s: 10.0,
            back_speed_mm_s: 6.0,
            walk_half_hz: 1.5,
            steer_gain_rad_s: 8.0,
            escape_speed_mm_s: 30.0,
            // measured escape-drive noise: idle ~14 Hz, p99.9 ~29 Hz at
            // 100 ms sampling (higher at tick level); a 15 Hz margin sat
            // inside the tail and still fired ~1.8 s of false escape per
            // 30 s run. 30 puts the trigger clear of the noise ceiling;
            // a genuine giant-fiber burst clears it by an order of magnitude.
            escape_burst_hz: 30.0,
            escape_duration_ms: 250,
            odor_k: 12.0,
            steer_scale_hz: 150.0,
            cast_surge: true,
            sweep_amp_rad_s: 4.5,
            sweep_period_ms: 2500,
            surge_ms: 2500,
            cast_speed: 0.4,
            odor_superclass: 0,
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

    pub fn for_con(con: &Connectome) -> Self {
        // male vocabulary: cb_sensory; FlyWire female: sensory
        let odor_superclass = con
            .superclass_names
            .iter()
            .position(|n| n == "cb_sensory" || n == "sensory")
            .expect("olfactory sensory superclass") as u8;
        WorldConfig {
            odor_superclass,
            ..WorldConfig::defaults()
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

fn default_zero() -> f64 {
    0.0
}

/// Typed full-embodiment snapshot. Wire-compatible with the historical
/// serde_json::Value layout (same keys, same nesting), so checkpoints
/// written by earlier versions still restore; the escape-idle and
/// concentration-dynamics states postdate that layout and default to 0.
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
    #[serde(default = "default_zero")]
    pub esc_idle: f64,
    #[serde(default = "default_zero")]
    pub conc_ema: f64,
    #[serde(default = "default_zero")]
    pub conc_peak: f64,
    #[serde(default)]
    pub cs: CastSurgeCp,
}

/// Checkpoint mirror of [`CastSurge`] (newer than the historical layout;
/// defaults keep old checkpoints parseable).
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CastSurgeCp {
    #[serde(default)]
    pub cast_until: u64,
    #[serde(default)]
    pub cast_start: u64,
    #[serde(default)]
    pub surge_until: u64,
    #[serde(default = "default_zero")]
    pub best_diff: f64,
    #[serde(default = "default_zero")]
    pub entry_conc: f64,
    #[serde(default = "default_zero")]
    pub best_bearing: f64,
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
    /// Slow EMA of the escape drive: the idle estimate the burst trigger
    /// compares against (see WorldConfig::escape_burst_hz).
    esc_idle: f64,
    /// Mean antenna concentration, EMA tau 300 ms (plant timescale).
    conc_ema: f64,
    /// Recent concentration peak: instant attack, decay tau 4 s. Losing a
    /// peak >2x the current level triggers casting (local search).
    conc_peak: f64,
    /// Cast-and-surge macro-state.
    cs: CastSurge,
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
            esc_idle: 0.0,
            conc_ema: 0.0,
            conc_peak: 0.0,
            cs: CastSurge::default(),
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
        // NOTE: a bodyId can appear as TWO steering instances (odor and
        // visual sets overlap); stats then yield the SAME EMA rate twice.
        // Overwrite per body — never accumulate — or shared bodies would
        // get base = 2x the measured rate and the static bias would
        // invert instead of nulling (review finding, 2026-09-17).
        let mut sum: HashMap<u64, f64> = HashMap::new();
        let mut samples = 0u64;
        for t in 0..measure_ms {
            self.step();
            if t % 100 == 0 {
                for (body, hz) in self.steer_instance_stats() {
                    sum.insert(body, hz);
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
            esc_idle: self.esc_idle,
            conc_ema: self.conc_ema,
            conc_peak: self.conc_peak,
            cs: CastSurgeCp {
                cast_until: self.cs.cast_until,
                cast_start: self.cs.cast_start,
                surge_until: self.cs.surge_until,
                best_diff: self.cs.best_diff,
                entry_conc: self.cs.entry_conc,
                best_bearing: self.cs.best_bearing,
            },
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
        self.esc_idle = cp.esc_idle;
        self.conc_ema = cp.conc_ema;
        self.conc_peak = cp.conc_peak;
        self.cs = CastSurge {
            cast_until: cp.cs.cast_until,
            cast_start: cp.cs.cast_start,
            surge_until: cp.cs.surge_until,
            best_diff: cp.cs.best_diff,
            entry_conc: cp.cs.entry_conc,
            best_bearing: cp.cs.best_bearing,
        };
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

        // escape idle estimate: slow EMA (tau ~4 s) of the escape drive,
        // tracked across all states so the burst trigger adapts to the
        // kernel's calibrated idle rate instead of a fixed constant.
        const ESC_IDLE_TAU: f64 = 4000.0;
        self.esc_idle += (d.escape - self.esc_idle) / (1.0 + ESC_IDLE_TAU);

        // 3. behavior -> motion; surge from mean antenna concentration
        let cl = self.field.conc(fx - self.cfg.antenna_offset_mm * h.sin(),
                                 fy + self.cfg.antenna_offset_mm * h.cos());
        let cr = self.field.conc(fx + self.cfg.antenna_offset_mm * h.sin(),
                                 fy - self.cfg.antenna_offset_mm * h.cos());
        let mean_conc = 0.5 * (cl + cr);
        // concentration dynamics for cast-and-surge: plant-time EMA plus a
        // decaying recent-peak envelope (attack instant, decay tau 4 s).
        const CONC_TAU: f64 = 300.0;
        const PEAK_DECAY_TAU: f64 = 4000.0;
        self.conc_ema += (mean_conc - self.conc_ema) / (1.0 + CONC_TAU);
        self.conc_peak = (self.conc_ema).max(self.conc_peak / (1.0 + 1.0 / PEAK_DECAY_TAU));
        let (v, w, label) = self.motion(d, mean_conc, cl, cr);

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
            // clamp to wall and steer toward center next ticks.
            // NOTE: the heading target is the direction from the fly TO
            // the center = atan2(-y, -x); atan2(y, x) points OUTWARD and
            // pinned clamped flies against the wall walking radially into
            // it (v > 0, zero net displacement) for entire runs.
            let scale = self.cfg.arena_radius_mm / r;
            self.fly.x = nx * scale;
            self.fly.y = ny * scale;
            let to_center = (-self.fly.y).atan2(-self.fly.x);
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

    fn motion(&mut self, d: Drives, mean_conc: f64, cl: f64, cr: f64) -> (f64, f64, &'static str) {
        if self.t_ms < self.escape_until {
            return (self.cfg.escape_speed_mm_s, 0.0, "escape");
        }
        // re-arm only after the burst excursion decays back toward idle
        if d.escape < self.esc_idle + 0.5 * self.cfg.escape_burst_hz {
            self.escape_armed = true;
        }
        if self.escape_armed && escape_is_burst(d.escape, self.esc_idle, self.cfg.escape_burst_hz)
        {
            self.escape_until = self.t_ms + self.cfg.escape_duration_ms;
            self.escape_armed = false;
            return (self.cfg.escape_speed_mm_s, 0.0, "escape");
        }
        let walk_n = d.walk / (d.walk + self.cfg.walk_half_hz);
        let back_n = d.walk_back / (d.walk_back + self.cfg.walk_half_hz);
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
        if self.groom_ema > self.cfg.groom_threshold_hz && walk_n < 0.3 && self.conc_ema < 0.1 {
            return (0.0, 0.0, "groom");
        }
        let surge = 1.0 + self.cfg.surge_gain * mean_conc.min(1.0);
        // Cast-and-surge: with cast_surge enabled, the cast->surge cycle IS
        // the locomotion controller. Casts sample the gradient across
        // bearings (concentration-anchored, kernel-noise-free); surges run
        // at the anchored bearing and abort into a cast on plume loss
        // (local search). The instance-level steering decode (signal +4 Hz
        // vs noise sigma ~100-300 Hz at this plant — see
        // WorldConfig::cast_surge) runs only when the machine is disabled.
        if self.cfg.cast_surge {
            // --- state transitions ---
            if self.cs.in_cast(self.t_ms) && self.cs.re_found(self.conc_ema) {
                // plume re-found mid-sweep: surge immediately
                self.cs.exit_to_surge(self.t_ms, self.fly.heading, &self.cfg);
            } else if self.cs.cast_until != 0
                && !self.cs.in_cast(self.t_ms)
                && !self.cs.in_surge(self.t_ms)
            {
                // cast expired without re-finding: anchor best bearing, surge
                self.cs.exit_to_surge(self.t_ms, self.fly.heading, &self.cfg);
            } else if self.cs.in_surge(self.t_ms)
                && (self.conc_ema >= CAST_STAY_CONC
                    || self.cs.peak_lost(self.conc_ema, self.conc_peak))
            {
                // plume core reached (arrived: switch to a tight local
                // loop) or plume lost mid-surge (overshoot) — either way
                // the surge ends early into a cast
                self.cs.force_cast(self.t_ms, self.conc_ema, &self.cfg);
            } else if !self.cs.in_cast(self.t_ms) && !self.cs.in_surge(self.t_ms) {
                // fresh start or surge expired: probe cast (re-anchor)
                self.cs.force_cast(self.t_ms, self.conc_ema, &self.cfg);
            }
            // --- dispatch ---
            if self.cs.in_cast(self.t_ms) {
                let mut w_cmd =
                    self.cs.cast_step(self.t_ms, self.fly.heading, cl, cr, &self.cfg);
                // Second half of the sweep, anchor found: blend the sweep
                // with a gentle turn onto the anchored bearing so the cast's
                // own slow drift creeps up-gradient instead of wandering
                // randomly (the anchor otherwise only pays off on the NEXT
                // surge).
                if !self.cs.best_bearing.is_nan()
                    && self.t_ms >= self.cs.cast_start + self.cfg.sweep_period_ms / 2
                {
                    w_cmd = 0.5 * w_cmd + self.cs.surge_w(self.fly.heading).clamp(-1.0, 1.0);
                }
                let v = self.cfg.walk_speed_mm_s * walk_n * self.cfg.cast_speed;
                let label = if walk_n > 0.3 { "walk" } else { "rest" };
                return (v, w_cmd, label);
            }
            let w_cmd = self.cs.surge_w(self.fly.heading);
            let v =
                (self.cfg.walk_speed_mm_s * walk_n - self.cfg.back_speed_mm_s * back_n) * surge;
            let label = if back_n > 0.5 && back_n > walk_n {
                "walk_back"
            } else if walk_n > 0.3 {
                "walk"
            } else {
                "rest"
            };
            return (v, w_cmd, label);
        }
        let v = (self.cfg.walk_speed_mm_s * walk_n - self.cfg.back_speed_mm_s * back_n) * surge;
        let w = self.cfg.steer_gain_rad_s * (d.steer / self.cfg.steer_scale_hz).tanh();
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

    /// None when the (gitignored) male artifact is absent — e.g. a fresh
    /// CI checkout — so artifact-dependent tests skip instead of failing.
    fn make_sim() -> Option<Sim> {
        let path = Path::new("../../bridge/out/connectome-malecns-v1.0");
        if !path.is_dir() {
            eprintln!("skipping: artifact required for world tests");
            return None;
        }
        let con = Some(Connectome::load(path).unwrap());
        let params: brain::LifParams = serde_json::from_slice(
            &std::fs::read("../../params/homeostasis_v1.json").unwrap(),
        )
        .unwrap_or_else(|_| brain::LifParams::default_for(con.as_ref().unwrap()));
        let map: transduce::BehaviorMapFile = serde_json::from_slice(
            &std::fs::read("../../scenarios/behavior_map.json").unwrap(),
        )
        .unwrap();
        let field = transduce::OdorField { x: 25.0, y: 0.0, strength: 1.0, radius_mm: 8.0 };
        let cfg = WorldConfig { seed: 7, ..WorldConfig::for_con(con.as_ref().unwrap()) };
        let mut sim = Sim::new(StdArc::new(con.unwrap()), params, &map, field, cfg);
        if let Ok(insts) =
            serde_json::from_slice::<Vec<transduce::SteerInstance>>(&std::fs::read(
                "../../scenarios/steer_instances.json",
            )
            .unwrap())
        {
            sim.apply_steer_instances(&insts);
        }
        Some(sim)
    }

    /// Backward compatibility: the historical serde_json::Value layout
    /// (tick at top level, brain without its tick key, no escape-idle /
    /// concentration-dynamics keys) must still parse and restore. The
    /// missing new-state keys default to 0, so continuation from an OLD
    /// checkpoint is deterministic but not bit-identical to the writing
    /// binary's dynamics (state the old format could not carry);
    /// bit-identical resume for the current shape is covered by
    /// [`tests::resume_is_bit_identical`].
    #[test]
    fn restore_legacy_checkpoint_shape() {
        let Some(mut sim) = make_sim() else { return; };
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
        let Some(mut sim2) = make_sim() else { return; };
        sim2.restore(&legacy);
        let Some(mut sim3) = make_sim() else { return; };
        sim3.restore(&legacy);
        for _ in 0..300 {
            let a = sim2.step();
            let b = sim3.step();
            assert_eq!(a.fly, b.fly, "legacy restore is not deterministic");
        }
    }

    /// Wall clamp must return a clamped fly INTO the arena. The historical
    /// bug: `to_center = atan2(y, x)` pointed outward, so a fly walking
    /// into the wall was steered radially back into it and sat pinned at
    /// r = arena_radius with v > 0 (zero net displacement) for whole runs.
    #[test]
    fn wall_clamp_returns_fly_to_center() {
        let Some(mut sim) = make_sim() else { return; };
        sim.fly = Fly { x: 39.0, y: 0.0, heading: 0.0 }; // walking outward
        let mut min_r = 40.0f64;
        for _ in 0..10000 {
            sim.step();
            min_r = min_r.min((sim.fly.x * sim.fly.x + sim.fly.y * sim.fly.y).sqrt());
        }
        assert!(min_r < 39.0, "clamped fly never re-entered: min r {min_r}");
    }

    /// Acceptance: checkpoint -> continue == uninterrupted run, bit-identical.
    #[test]
    fn resume_is_bit_identical() {        let Some(mut sim) = make_sim() else { return; };
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
        let Some(mut sim2) = make_sim() else { return; };
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

/// Escape burst test: the drive must exceed the slow idle EMA by the
/// configured margin (an absolute threshold cannot separate idle noise
/// from a response under global calibration).
fn escape_is_burst(escape_hz: f64, idle_hz: f64, burst_hz: f64) -> bool {
    escape_hz > idle_hz + burst_hz
}

/// Minimum recent peak for the plume-loss abort to arm.
const CAST_PEAK_FLOOR: f64 = 0.15;
/// Concentration rise over the cast-entry level (factor) that aborts a
/// cast early (plume re-found -> surge immediately).
const CAST_REFIND: f64 = 1.5;
/// Concentration above which the fly has reached the plume core: surges
/// abort into casts (tight local loop) instead of running their full
/// length straight through the source.
const CAST_STAY_CONC: f64 = 0.55;
/// Surge alignment turn-rate cap (rad/s).
const SURGE_ALIGN_RAD_S: f64 = 2.0;

/// Cast-and-surge macro-state (see [`WorldConfig::cast_surge`]).
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CastSurge {
    cast_until: u64,
    cast_start: u64,
    surge_until: u64,
    best_diff: f64,
    /// Concentration at cast entry: a significant RISE above this (not a
    /// fraction of the historical peak, which can equal the entry level)
    /// means the plume was re-found mid-sweep.
    entry_conc: f64,
    /// Heading (rad) at the highest sampled concentration during the cast.
    best_bearing: f64,
}

impl CastSurge {
    fn in_cast(&self, t: u64) -> bool {
        t < self.cast_until
    }

    fn in_surge(&self, t: u64) -> bool {
        t < self.surge_until
    }

    /// Plume lost during a surge: the concentration has fallen below half
    /// a recent peak worth arming (the fly overshot a rewarding spot and
    /// should cast — local search — instead of walking on).
    fn peak_lost(&self, conc_ema: f64, conc_peak: f64) -> bool {
        conc_peak > CAST_PEAK_FLOOR && conc_ema < 0.5 * conc_peak
    }

    /// Unconditional cast entry: probe/re-anchor cycle during approach
    /// (the surge re-casts every surge_ms to refresh its bearing) and the
    /// peak-loss abort from a surge.
    fn force_cast(&mut self, t: u64, conc_ema: f64, cfg: &WorldConfig) {
        self.cast_start = t;
        self.cast_until = t + cfg.sweep_period_ms;
        self.surge_until = 0;
        self.entry_conc = conc_ema;
        self.best_diff = f64::NEG_INFINITY;
        self.best_bearing = f64::NAN; // anchored on the first cast_step
    }

    /// Cast steering command (sinusoidal heading sweep) and the
    /// concentration-anchored bearing record. The anchor is the LEFT-RIGHT
    /// antenna DIFFERENCE, not the mean and not either antenna alone:
    /// - the bilateral mean is CONSTANT on a fixed-position sweep (the
    ///   antennas trace a ~offset circle in a radially-symmetric field) —
    ///   it carries no bearing information;
    /// - a single antenna's level is dominated by the cast's own
    ///   translation (drifting 6 mm radially changes conc ~5x more than
    ///   the 0.6 mm bearing signal);
    /// - cl(phi) - cr(phi) = 2*offset*|grad|*cos(phi - phi_grad + pi/2)
    ///   is translation-invariant (only the slow |grad| scale changes
    ///   with drift): its max sits at heading phi_grad - pi/2, i.e. the
    ///   body bearing whose left antenna points along the gradient.
    fn cast_step(&mut self, t: u64, heading: f64, cl: f64, cr: f64, cfg: &WorldConfig) -> f64 {
        let phase = 2.0 * std::f64::consts::PI
            * (t - self.cast_start) as f64
            / cfg.sweep_period_ms as f64;
        let diff = cl - cr;
        if diff > self.best_diff || self.best_bearing.is_nan() {
            self.best_diff = diff;
            self.best_bearing = heading;
        }
        cfg.sweep_amp_rad_s * phase.sin()
    }

    /// Concentration re-found during a cast: rose CAST_REFIND-fold over
    /// the entry level. In clean air (entry 0) a cast always runs its
    /// full course — there is nothing to re-find.
    fn re_found(&self, conc_ema: f64) -> bool {
        self.entry_conc > 1e-9 && conc_ema >= CAST_REFIND * self.entry_conc
    }

    /// Anchor the surge heading. `best_bearing` is the heading at which
    /// the LEFT antenna sampled its maximum — but the left antenna sits
    /// 90 deg CCW of heading, so that heading points the antenna (not the
    /// body) at the gradient: the movement bearing is +90 deg from it.
    /// (argmax_phi cl = phi_grad - pi/2; surge heading = phi_grad.)
    fn anchor_surge_bearing(&mut self, heading: f64) {
        if self.best_bearing.is_nan() {
            self.best_bearing = heading;
        }
        self.best_bearing = (self.best_bearing + 0.5 * std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI);
    }

    fn exit_to_surge(&mut self, t: u64, heading: f64, cfg: &WorldConfig) {
        self.cast_until = 0;
        self.surge_until = t + cfg.surge_ms;
        self.anchor_surge_bearing(heading);
    }

    /// Surge steering: rate-limited alignment onto the recorded bearing,
    /// then straight.
    fn surge_w(&self, heading: f64) -> f64 {
        let diff = (self.best_bearing - heading + std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        if diff.abs() < 0.1 {
            0.0
        } else {
            diff.clamp(-SURGE_ALIGN_RAD_S, SURGE_ALIGN_RAD_S)
        }
    }
}

#[cfg(test)]
mod pure_tests {
    use super::*;

    #[test]
    fn cast_surge_cycle_transitions() {
        let cfg = WorldConfig::defaults();
        let mut cs = CastSurge::default();
        // fresh start: probe cast
        cs.force_cast(0, 0.0, &cfg);
        assert!(cs.in_cast(1));
        // cast runs one sweep period, then expires -> anchor + surge
        let t_end = cfg.sweep_period_ms;
        assert!(cs.in_cast(t_end - 1));
        assert!(!cs.in_cast(t_end));
        cs.exit_to_surge(t_end, 0.0, &cfg);
        assert!(cs.in_surge(t_end + 1));
        // surge expires -> probe cast again (continuous re-anchor cycle)
        let t_surge_end = t_end + cfg.surge_ms;
        assert!(cs.in_surge(t_surge_end - 1));
        cs.force_cast(t_surge_end, 0.0, &cfg);
        assert!(cs.in_cast(t_surge_end + 1));
    }

    #[test]
    fn surge_aborts_on_plume_loss() {
        let cfg = WorldConfig::defaults();
        let mut cs = CastSurge::default();
        cs.force_cast(0, 0.6, &cfg);
        cs.exit_to_surge(2000, 0.0, &cfg);
        assert!(cs.in_surge(2001));
        // concentration still near its peak: keep surging
        assert!(!cs.peak_lost(0.5, 0.6));
        // concentration collapsed below half the recent peak: local search
        assert!(cs.peak_lost(0.25, 0.6));
        // a trivial peak never arms the abort
        assert!(!cs.peak_lost(0.0, 0.1));
    }

    #[test]
    fn cast_anchors_bearing_at_max_conc_then_surges() {
        let cfg = WorldConfig::defaults();
        let mut cs = CastSurge::default();
        cs.force_cast(0, 0.1, &cfg);
        // sweep: left-right antenna difference highest when heading = 1.0
        let _ = cs.cast_step(100, 0.2, 0.1, 0.1, &cfg);
        let _ = cs.cast_step(200, 1.0, 0.5, 0.1, &cfg);
        let _ = cs.cast_step(300, 2.0, 0.3, 0.3, &cfg);
        assert_eq!(cs.best_bearing, 1.0);
        // cast ends -> the surge heading is +90 deg from the antenna-optimal
        // heading (the left antenna rides 90 deg CCW of the body axis), so
        // the fly moves along the gradient, not perpendicular to it
        cs.exit_to_surge(2 * cfg.sweep_period_ms + 1, 0.0, &cfg);
        let target = 1.0 + 0.5 * std::f64::consts::PI;
        assert_eq!(cs.surge_w(target), 0.0, "aligned: no steering");
        let w = cs.surge_w(1.0);
        assert!(w > 0.0 && w <= 2.0 + 1e-9, "must turn CCW toward target");
        // opposite side turns negative
        assert!(cs.surge_w(target + std::f64::consts::PI) < 0.0);
    }

    #[test]
    fn cast_aborts_only_on_real_concentration_rise() {
        let cfg = WorldConfig::defaults();
        let mut cs = CastSurge::default();
        // probe cast entered at conc 0.2 (after an overshoot)
        cs.force_cast(0, 0.2, &cfg);
        assert!(cs.in_cast(1));
        // staying at the entry level must NOT count as re-found (the
        // historical fraction-of-peak check exited instantly here)
        assert!(!cs.re_found(0.2));
        // a 1.5x rise over entry does
        assert!(cs.re_found(0.31));
        cs.exit_to_surge(10, 0.5, &cfg);
        assert!(!cs.in_cast(11) && cs.in_surge(11));
        // clean-air cast (entry 0) never re-finds: runs full course
        let mut cs3 = CastSurge::default();
        cs3.force_cast(0, 0.0, &cfg);
        assert!(!cs3.re_found(0.4));
    }

    #[test]
    fn escape_burst_needs_excursion_over_idle() {
        // idle 10, burst margin 15: trigger at 25+ never at noise level
        assert!(!escape_is_burst(24.9, 10.0, 15.0));
        assert!(escape_is_burst(25.1, 10.0, 15.0));
        // a globally-shifted idle (calibration change) does NOT fire:
        // what matters is the excursion, not the absolute rate
        assert!(!escape_is_burst(24.9, 20.0, 15.0));
        assert!(escape_is_burst(40.0, 20.0, 15.0));
    }
}
