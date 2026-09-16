//! Headless optomotor drum assay: 12 bright stripes on a ring of radius
//! 20 mm rotate about the arena center at +-omega rad/s; a fly showing the
//! optomotor response turns with the drum. The binary runs three
//! conditions itself (+omega, -omega, static control; same seed), writes
//! `<out>-{plus,minus,static}/summary.json` (+ heading trace each), and
//! prints a final verdict: each moving condition must turn in the drum's
//! direction with |omega_fly| above the static control.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use brain::LifParams;
use connectome::Connectome;
use serde_json::json;
use transduce::{BehaviorMapFile, OdorField, SteerInstance, VisObject};
use world::{Sim, WorldConfig};

const DRUM_RADIUS_MM: f64 = 20.0;
const DRUM_OBJECTS: usize = 12;
/// Stripe radius and brightness (visible against a dark arena).
const STRIPE_RADIUS_MM: f64 = 2.0;
const STRIPE_BRIGHTNESS: f64 = 1.0;
/// Warm up 2 s before recording (EMA settling), as in `sim_headless`.
const WARMUP_MS: u64 = 2000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args.get(1).map(|s| s.as_str()).unwrap_or("bridge/out/connectome-malecns-v1.0");
    let params_path = args.get(2).map(|s| s.as_str()).unwrap_or("params/homeostasis_v1.json");
    let map_path = args.get(3).map(|s| s.as_str()).unwrap_or("scenarios/behavior_map.json");
    let out_prefix = args.get(4).map(|s| s.as_str()).unwrap_or("runs/optomotor");
    let seconds: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(30);
    let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(7);
    // drum speed magnitude; the assay runs +omega, -omega and 0
    let drum_omega: f64 = args
        .get(7)
        .and_then(|s| s.parse::<f64>().ok())
        .map(f64::abs)
        .unwrap_or(0.5);

    let con = Arc::new(Connectome::load(Path::new(con_dir)).expect("load connectome"));
    let mut params: LifParams =
        serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    params.seed = seed;
    if params.gains_exc.len() != con.superclass_names.len() {
        params = LifParams::default_for(&con);
    }
    let map: BehaviorMapFile =
        serde_json::from_slice(&fs::read(map_path).unwrap()).unwrap();
    // both steering files are optional in this assay
    let steer_insts: Vec<SteerInstance> = fs::read("scenarios/steer_instances.json")
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let vis_steer: Vec<SteerInstance> = fs::read("scenarios/visual_steer_instances.json")
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();

    // run all three conditions first: the verdict needs the static control
    let conds = [("plus", drum_omega), ("minus", -drum_omega), ("static", 0.0)];
    let mut results: Vec<(&str, f64, f64)> = Vec::new();
    for (tag, omega) in conds {
        let out_dir = format!("{}-{}", out_prefix, tag);
        let fly_omega = run_condition(
            &con, &params, &map, &steer_insts, &vis_steer, omega, seconds, seed, &out_dir,
        );
        results.push((tag, omega, fly_omega));
    }

    // Verdict: each moving condition must turn in the drum's direction and
    // harder than the static control (same seed).
    let static_abs = results
        .iter()
        .find(|(tag, ..)| *tag == "static")
        .map(|(_, _, w)| w.abs())
        .unwrap_or(0.0);
    let mut all_pass = true;
    for (tag, omega, fly_omega) in &results {
        let status = if *omega == 0.0 {
            "control".to_string()
        } else {
            let ok = fly_omega.signum() == omega.signum() && fly_omega.abs() > static_abs;
            all_pass &= ok;
            if ok { "PASS" } else { "FAIL" }.to_string()
        };
        println!(
            "drum {:+.3} rad/s -> fly {:+.4} rad/s [{}] (wrote {}-{})",
            omega, fly_omega, status, out_prefix, tag
        );
    }
    println!(
        "optomotor verdict: {}",
        if all_pass { "PASS" } else { "FAIL" }
    );
}

/// One drum condition: fresh sim, warmup, then record fly heading every
/// 100 ms; writes trace.json + summary.json and returns the mean angular
/// velocity (rad/s) over the recorded span.
#[allow(clippy::too_many_arguments)]
fn run_condition(
    con: &Arc<Connectome>,
    params: &LifParams,
    map: &BehaviorMapFile,
    steer_insts: &[SteerInstance],
    vis_steer: &[SteerInstance],
    drum_omega: f64,
    seconds: u64,
    seed: u64,
    out_dir: &str,
) -> f64 {
    let cfg = WorldConfig {
        seed,
        ..WorldConfig::for_con(con)
    };
    // no odor in this assay: field parked far away at zero strength
    let field = OdorField { x: 1.0e6, y: 1.0e6, strength: 0.0, radius_mm: 8.0 };
    let mut sim = Sim::new(Arc::clone(con), params.clone(), map, field, cfg);
    sim.apply_steer_instances(steer_insts);
    sim.add_steer_instances(vis_steer);
    sim.vis_objects = drum(drum_omega);

    for _ in 0..WARMUP_MS {
        sim.step();
    }

    let total_ticks = seconds * 1000;
    let sample_every = 100; // ms
    let mut trace: Vec<(u64, f64)> =
        Vec::with_capacity((total_ticks / sample_every + 1) as usize);
    for t in 0..total_ticks {
        let info = sim.step();
        if t % sample_every == 0 {
            trace.push((info.t_ms, info.fly.heading));
        }
    }

    // Mean angular velocity: sum of wrapped successive heading deltas over
    // the recorded span. Per-sample deltas stay far below pi, so this is
    // immune to multi-revolution wrapping of the absolute heading.
    let mut net = 0.0f64;
    for w in trace.windows(2) {
        net += wrap_pi(w[1].1 - w[0].1);
    }
    let dur_s = match (trace.first(), trace.last()) {
        (Some(a), Some(b)) if b.0 > a.0 => (b.0 - a.0) as f64 / 1000.0,
        _ => 0.0,
    };
    let fly_omega = if dur_s > 0.0 { net / dur_s } else { 0.0 };

    fs::create_dir_all(out_dir).unwrap();
    let trace_json: Vec<serde_json::Value> = trace
        .iter()
        .map(|&(t_ms, heading)| json!({"t_ms": t_ms, "heading": heading}))
        .collect();
    fs::write(
        Path::new(out_dir).join("trace.json"),
        serde_json::to_vec(&json!(trace_json)).unwrap(),
    )
    .unwrap();
    fs::write(
        Path::new(out_dir).join("summary.json"),
        serde_json::to_vec_pretty(&json!({
            "drum_omega": drum_omega,
            "fly_omega": fly_omega,
            "seconds": seconds,
        }))
        .unwrap(),
    )
    .unwrap();
    fly_omega
}

/// Stripe ring: `DRUM_OBJECTS` bright stripes evenly spaced by angle on a
/// circle of `DRUM_RADIUS_MM` about the arena center, rotating at
/// `omega` rad/s (signed; + = CCW).
fn drum(omega: f64) -> Vec<VisObject> {
    (0..DRUM_OBJECTS)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / DRUM_OBJECTS as f64;
            VisObject {
                x: DRUM_RADIUS_MM * a.cos(),
                y: DRUM_RADIUS_MM * a.sin(),
                radius_mm: STRIPE_RADIUS_MM,
                brightness: STRIPE_BRIGHTNESS,
                pivot: (0.0, 0.0),
                ang_vel_rad_s: omega,
                ang0: a,
            }
        })
        .collect()
}

/// Wrap an angle difference to (-pi, pi].
fn wrap_pi(d: f64) -> f64 {
    (d + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}
