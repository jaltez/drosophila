//! Sensory transducers and motor decode.
//!
//! - [`DnArbiter`]: DN firing rates (EMA'd) -> behavior drives. Walk /
//!   escape / feed / groom come from type-level votes
//!   (`scenarios/behavior_map.json`); steering comes from instance-level
//!   weights (`scenarios/steer_instances.json`), produced by the
//!   per-instance `dn_screen` run: each DN neuron's response asymmetry to
//!   left- vs right-antenna odor drive becomes a signed steering weight
//!   (+ = prefers left-antenna drive). The bilateral comparison over ~130
//!   instances is antisymmetric by construction and yields chemotactic
//!   turning toward the higher-concentration side.
//! - [`OdorField`]: point-source odor concentration; per-antenna sampling
//!   gives the left/right asymmetry that drives the steering instances.

use std::collections::HashMap;

use connectome::Connectome;
use serde::Deserialize;

/// EMA window for DN rates (ms) — behavior-relevant timescale.
const RATE_TAU_MS: f64 = 200.0;
const ALPHA: f64 = 1.0 / (1.0 + RATE_TAU_MS); // per-tick EMA factor
/// High-pass window for the anatomical steering sum (ms): removes the DC
/// so steering responds to bearing CHANGES (cast-and-surge dynamics).
const STEER_DC_TAU_MS: f64 = 2000.0;
const ALPHA_DC: f64 = 1.0 / (1.0 + STEER_DC_TAU_MS);

#[derive(Debug, Deserialize)]
pub struct BehaviorMapFile {
    pub votes_per_type: HashMap<String, VoteDef>,
    pub behaviors: Vec<String>,
    #[serde(default)]
    pub arbiter: ArbiterCfg,
}

#[derive(Debug, Deserialize)]
pub struct VoteDef {
    pub behavior: String,
    pub w: f64,
}

#[derive(Debug, Default, Deserialize)]
pub struct ArbiterCfg {
    #[serde(default = "default_hysteresis")]
    pub hysteresis_ticks: u32,
    #[serde(default)]
    pub bias: HashMap<String, f64>,
}

fn default_hysteresis() -> u32 {
    500
}

/// One steering instance from `dn_screen` output.
#[derive(Debug, Deserialize)]
pub struct SteerInstance {
    #[serde(rename = "bodyId")]
    pub body_id: u64,
    pub w: f64,
    /// Baseline (no-odor) firing rate (Hz); contribution = w * (rate - base).
    #[serde(default)]
    pub base: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Drives {
    pub walk: f64,
    pub walk_back: f64,
    /// Signed steering drive (Hz-equivalent): positive = turn left (CCW).
    pub steer: f64,
    pub escape: f64,
    pub feed: f64,
    pub groom: f64,
}

/// Tracks DN population rates and converts them to behavior drives.
pub struct DnArbiter {
    /// DN neuron indices (into connectome), ascending.
    dn_idx: Vec<u32>,
    /// EMA of spikes-per-tick per DN instance (rate_hz = ema * 1000).
    ema: Vec<f64>,
    /// type code per DN instance.
    type_of: Vec<u16>,
    /// (type code, behavior, weight) votes for non-steer behaviors.
    votes: Vec<(u16, String, f64)>,
    /// Instance-level steering weights: (dn slot, signed weight).
    /// Instance-level steering weights: (dn slot, signed weight, base Hz).
    steer_instances: Vec<(usize, f64, f64)>,
    /// Slow EMA of the raw steering sum (DC estimate), subtracted in drives().
    steer_dc: f64,
    /// Slow EMA of the raw steering sum (DC offset estimate).
    /// instances per voted type, for mean-rate normalization.
    instances_per_type: HashMap<u16, usize>,
}

impl DnArbiter {
    pub fn new(con: &Connectome, map: &BehaviorMapFile) -> Self {
        let mut dn_idx = Vec::new();
        let mut type_of = Vec::new();
        for i in 0..con.func_classes.len() as u32 {
            if con.func_classes[i as usize] == connectome::FUNC_DN {
                dn_idx.push(i);
                type_of.push(con.type_codes[i as usize]);
            }
        }
        let mut votes = Vec::new();
        let mut instances_per_type = HashMap::new();
        for (name, def) in &map.votes_per_type {
            if def.behavior == "steer" {
                continue; // steering is instance-level
            }
            if let Some(code) = con.type_index(name) {
                votes.push((code, def.behavior.clone(), def.w));
                let n = con.type_codes.iter().filter(|&&c| c == code).count();
                instances_per_type.insert(code, n.max(1));
            }
        }
        let n_dn = dn_idx.len();
        DnArbiter {
            dn_idx,
            ema: vec![0.0; n_dn],
            type_of,
            votes,
            steer_instances: Vec::new(),
            steer_dc: 0.0,
            instances_per_type,
        }
    }

    /// Set instance-level steering weights (bodyId -> signed weight,
    /// + = prefers left-antenna drive). Silently skips bodyIds that are
    /// not DN instances of this connectome.
    pub fn set_steer_instances(&mut self, con: &Connectome, insts: &[SteerInstance]) {
        let mut out = Vec::new();
        for inst in insts {
            if let Some(idx) = con.index_of_body(inst.body_id) {
                if let Ok(k) = self.dn_idx.binary_search(&idx) {
                    out.push((k, inst.w, inst.base));
                }
            }
        }
        self.steer_instances = out;
    }

    pub fn steer_instance_count(&self) -> usize {
        self.steer_instances.len()
    }

    /// Feed one tick's spikes (any slice; non-DN spikes ignored).
    pub fn observe(&mut self, spikes: &[u32]) {
        for e in self.ema.iter_mut() {
            *e *= 1.0 - ALPHA;
        }
        for &s in spikes {
            if let Ok(k) = self.dn_idx.binary_search(&s) {
                self.ema[k] += ALPHA;
            }
        }
        self.steer_dc += (self.steer_drive() - self.steer_dc) * ALPHA_DC;
    }

    /// Mean EMA rate (Hz) across instances of a DN type, 0 if absent.
    pub fn type_rate_hz(&self, type_code: u16) -> f64 {
        let n = match self.instances_per_type.get(&type_code) {
            Some(&n) => n,
            None => return 0.0,
        };
        let mut sum = 0.0f64;
        for (e, &t) in self.ema.iter().zip(&self.type_of) {
            if t == type_code {
                sum += e;
            }
        }
        sum * 1000.0 / n as f64
    }

    /// Signed steering drive: weighted sum over screen-selected DN
    /// instances. Positive = turn left (CCW), toward a left-side source.
    fn steer_drive(&self) -> f64 {
        let mut s = 0.0;
        for &(k, w, base) in &self.steer_instances {
            s += w * (self.ema[k] * 1000.0 - base);
        }
        s
    }

    /// Aggregate behavior drives from current EMA rates.
    pub fn drives(&self) -> Drives {
        let mut d = Drives {
            steer: self.steer_drive() - self.steer_dc,
            ..Default::default()
        };
        for &(code, ref beh, w) in &self.votes {
            let rate = self.type_rate_hz(code) * w;
            match beh.as_str() {
                "walk" => d.walk += rate,
                "walk_back" => d.walk_back += rate,
                "escape" => d.escape += rate,
                "feed" => d.feed += rate,
                "groom" => d.groom += rate,
                _ => {}
            }
        }
        d
    }
}

/// Point-source odor field in a 2D arena (mm).
#[derive(Clone, Copy, Debug)]
pub struct OdorField {
    pub x: f64,
    pub y: f64,
    pub strength: f64,
    pub radius_mm: f64,
}

impl OdorField {
    /// Concentration at a point (arbitrary units, 0..=strength).
    pub fn conc(&self, px: f64, py: f64) -> f64 {
        let dx = px - self.x;
        let dy = py - self.y;
        self.strength / (1.0 + (dx * dx + dy * dy) / (self.radius_mm * self.radius_mm))
    }
}

/// Update per-sensory-neuron Poisson gains from odor geometry: left/right
/// antenna sample the field at lateral offsets; sided sensory neurons of
/// superclass `sup` get `1 + k * conc(antenna)`; unsided get the mean.
pub fn apply_odor_gains(
    con: &Connectome,
    brain_gains: &mut [f32],
    sup: u8,
    field: &OdorField,
    fly: (f64, f64, f64), // x, y, heading (rad, 0 = +x)
    antenna_offset_mm: f64,
    k: f64,
) {
    let (fx, fy, h) = fly;
    let (lx, ly) = (
        fx - antenna_offset_mm * h.sin(),
        fy + antenna_offset_mm * h.cos(),
    );
    let (rx, ry) = (
        fx + antenna_offset_mm * h.sin(),
        fy - antenna_offset_mm * h.cos(),
    );
    let cl = field.conc(lx, ly);
    let cr = field.conc(rx, ry);
    for i in 0..con.func_classes.len() {
        if con.func_classes[i] != connectome::FUNC_SENSORY
            || con.superclass_codes[i] != sup
        {
            continue;
        }
        let g = match con.side_codes[i] {
            1 => 1.0 + k * cl,
            2 => 1.0 + k * cr,
            _ => 1.0 + k * 0.5 * (cl + cr),
        };
        brain_gains[i] = g as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 7 neurons: 0-3 DNs (DNa01 L/R = type 1, DNg13 L/R = type 2),
    /// 4 inter, 5 sensory-left, 6 sensory-right.
    fn con_with_dns() -> Connectome {
        Connectome {
            dir: "syn".into(),
            manifest: serde_json::from_str(
                r#"{"name":"syn","counts":{"neurons":7,"edges":0,"synapses":0}}"#,
            )
            .unwrap(),
            body_ids: vec![101, 102, 103, 104, 105, 106, 107],
            func_classes: vec![2, 2, 2, 2, 0, 1, 1],
            superclass_codes: vec![0; 7],
            superclass_names: vec!["s".into()],
            side_codes: vec![1, 2, 1, 2, 0, 1, 2],
            type_codes: vec![1, 1, 2, 2, 0, 0, 0],
            type_names: vec!["other".into(), "DNa01".into(), "DNg13".into()],
            csr_offsets: vec![0; 8],
            csr_targets: vec![],
            csr_weights: vec![],
        }
    }

    fn map() -> BehaviorMapFile {
        serde_json::from_str(
            r#"{
                "votes_per_type": {
                    "DNa01": {"behavior": "walk", "w": 1.0}
                },
                "behaviors": ["rest", "walk"]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn arbiter_walk_and_instance_steer() {
        let con = con_with_dns();
        let mut arb = DnArbiter::new(&con, &map());
        // steer instances: neuron 0 (left) +2, neuron 3 (right) -2
        arb.set_steer_instances(
            &con,
            &[
                SteerInstance { body_id: 101, w: 2.0, base: 0.0 },
                SteerInstance { body_id: 104, w: -2.0, base: 0.0 },
            ],
        );
        assert_eq!(arb.steer_instance_count(), 2);

        // fire left-preferring instance every tick -> steer positive
        for _ in 0..5000 {
            arb.observe(&[0]);
        }
        let d = arb.drives();
        assert!(d.steer > 100.0, "steer {} (left-dominated > 0 after DC removal)", d.steer);
        // walk drive from DNa01 type mean over 2 instances (one firing)
        assert!(d.walk > 450.0, "walk {} Hz", d.walk);

        // switch to right-preferring instance -> steer flips negative
        for _ in 0..5000 {
            arb.observe(&[3]);
        }
        let d = arb.drives();
        assert!(d.steer < 0.0, "steer {} (right instance dominates)", d.steer);
    }

    #[test]
    fn odor_gains_are_sided() {
        let con = con_with_dns();
        let field = OdorField {
            x: 10.0,
            y: 0.0,
            strength: 1.0,
            radius_mm: 5.0,
        };
        // fly at origin heading +x; source dead ahead -> symmetric antennae
        let mut gains = vec![1.0f32; 7];
        apply_odor_gains(&con, &mut gains, 0, &field, (0.0, 0.0, 0.0), 0.6, 10.0);
        let expect = 1.0 + 10.0 * field.conc(0.0, 0.6);
        assert!((gains[5] - expect as f32).abs() < 1e-4, "{} vs {}", gains[5], expect);
        assert!((gains[5] - gains[6]).abs() < 1e-6);

        // heading +y: left antenna at (-0.6, 0), right at (+0.6, 0);
        // source at +x -> right antenna closer -> gains[6] > gains[5]
        let mut g2 = vec![1.0f32; 7];
        apply_odor_gains(
            &con,
            &mut g2,
            0,
            &field,
            (0.0, 0.0, std::f64::consts::FRAC_PI_2),
            0.6,
            10.0,
        );
        assert!(g2[6] > g2[5], "expected right-side gain larger");
        assert_eq!(g2[4], 1.0); // non-sensory untouched
    }
}
