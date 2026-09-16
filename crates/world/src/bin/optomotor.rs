//! Headless optomotor drum assay: 12 bright stripes on a ring of radius
//! 20 mm rotate about the arena center at +-omega rad/s; a fly showing the
//! optomotor response turns with the drum. The binary runs three
//! conditions itself (+omega, -omega, static control; same seed), writes
//! `<out>-{plus,minus,static}/summary.json` (+ heading trace each), and
//! prints a final verdict: each moving condition must turn in the drum's
//! direction with |omega_fly| above the static control.

use std::collections::HashMap;
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
    // onset-window length; the drum response adapts after a few seconds
    let seconds: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
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
    // per-instance centering (below) removes static bias; the global DC
    // high-pass would additionally absorb the sustained drum response —
    // optomotor following IS a DC behavior, so allow disabling the filter
    let mut map: BehaviorMapFile =
        serde_json::from_slice(&fs::read(map_path).unwrap()).unwrap();
    if let Ok(tau) = std::env::var("OPTOMOTOR_DC_TAU") {
        map.arbiter.steer_dc_tau_ms = tau.parse().unwrap_or(map.arbiter.steer_dc_tau_ms);
    }
    params.seed = seed;
    if params.gains_exc.len() != con.superclass_names.len() {
        params = LifParams::default_for(&con);
    }
    // both steering files are optional in this assay
    let steer_insts: Vec<SteerInstance> = fs::read("scenarios/steer_instances.json")
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let vis_steer: Vec<SteerInstance> = fs::read("scenarios/visual_steer_instances.json")
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();

    // Center per-instance bases against the static-scene response: the
    // diffusion-derived weights carry an intrinsic left/right imbalance
    // (previously -1.3 rad/s net turn) that swamps drum following.
    // Every instance's `base` is set to its measured static-scene rate,
    // nulling the static steering drive so only drum modulation remains.
    // diagnostic knobs: vis_sign flips the visual instance weights;
    // sets restricts which steering sets run (odor|vis|both)
    let vis_sign: f64 = std::env::var("OPTOMOTOR_VIS_SIGN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    // vis-only is the assay standard: the odor-steering set has no
    // stimulus here and its unaveraged noise (~1.9 rad/s spin) swamps the
    // visual response. Opt in with OPTOMOTOR_SETS=both|odor.
    let sets = std::env::var("OPTOMOTOR_SETS").unwrap_or_else(|_| "vis".into());
    let (only_odor, only_vis) = match sets.as_str() {
        "odor" => (true, false),
        "vis" => (false, true),
        _ => (false, false),
    };
    let mut steer_insts: Vec<SteerInstance> =
        if only_vis { Vec::new() } else { steer_insts };
    let mut vis_steer: Vec<SteerInstance> =
        if only_odor { Vec::new() } else { vis_steer };
    for i in vis_steer.iter_mut() {
        i.w *= vis_sign;
    }
    let (steer_insts, vis_steer, pre_bias) =
        centered_instances(&con, &params, &map, &steer_insts, &vis_steer, seed);
    println!(
        "centered {} + {} instances against static scene (pre-centering bias {:+.3} Hz-equiv); wrote {}-centered.json",
        steer_insts.len(),
        vis_steer.len(),
        pre_bias,
        out_prefix
    );

    // Open-loop channel characterization: freeze the fly at the arena
    // center, rotate the drum at several omegas, and record the mean
    // arbiter steering drive. Monotone increasing through zero = the
    // visual steering channel is sign-correct for following.
    if std::env::var("OPTOMOTOR_OPENLOOP").as_deref() == Ok("1") {
        let field = OdorField { x: 1.0e6, y: 1.0e6, strength: 0.0, radius_mm: 8.0 };
        let (c_odor, c_vis, _) =
            centered_instances(&con, &params, &map, &steer_insts, &vis_steer, seed);
        println!("omega_rad_s  mean_steer_Hz  (fly frozen; DC tau max)");
        for om in [-1.0f64, -0.5, -0.25, 0.25, 0.5, 1.0] {
            let cfg = WorldConfig {
                seed,
                freeze_body: true,
                ..WorldConfig::for_con(&con)
            };
            let mut sim = Sim::new(Arc::clone(&con), params.clone(), &map, field, cfg);
            sim.apply_steer_instances(&c_odor);
            sim.add_steer_instances(&c_vis);
            sim.vis_objects = drum(om);
            for _ in 0..WARMUP_MS {
                sim.step();
            }
            let mut sum = 0.0;
            let n = 5000.0;
            for _ in 0..n as u64 {
                sum += sim.step().drives.steer;
            }
            println!("{:+8.2}      {:+9.3}", om, sum / n);
        }
        return;
    }

    // Seed replication: single-seed drift noise (~0.4 rad/s) rivals the
    // drum response (~0.2), so the verdict averages OPTOMOTOR_SEEDS runs
    // per condition (default 1 for the trace files, more for a verdict).
    let n_seeds: u64 = std::env::var("OPTOMOTOR_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let conds = [("plus", drum_omega), ("minus", -drum_omega), ("static", 0.0)];
    let mut results: Vec<(&str, f64, f64)> = Vec::new();
    for (tag, omega) in conds {
        let out_dir = format!("{}-{}", out_prefix, tag);
        let mut total = 0.0;
        for k in 0..n_seeds {
            let sd = seed + k * 7919;
            // trace files from seed[0] only; later seeds just vote
            let dir = if k == 0 { out_dir.clone() } else { format!("{}.s{}", out_dir, sd) };
            total += run_condition(
                &con, &params, &map, &steer_insts, &vis_steer, omega, seconds, sd, &dir,
            );
        }
        results.push((tag, omega, total / n_seeds as f64));
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
            "drum {:+.3} rad/s -> fly {:+.4} rad/s (mean of {}) [{}] (wrote {}-{})",
            omega, fly_omega, n_seeds, status, out_prefix, tag
        );
    }
    println!(
        "optomotor verdict: {}",
        if all_pass { "PASS" } else { "FAIL" }
    );
}

/// Measure per-instance mean rates under a STATIC scene (same seed and
/// initial conditions as the assay conditions) and rewrite every
/// instance's `base` to that rate, so the static net steering drive is
/// exactly zero. Returns the centered (odor, visual) sets and the
/// pre-centering static bias (Hz-equivalent) for the report. The centered
/// union is also written to `<out_prefix>-centered.json` for reuse.
fn centered_instances(
    con: &Arc<Connectome>,
    params: &LifParams,
    map: &BehaviorMapFile,
    steer_insts: &[SteerInstance],
    vis_steer: &[SteerInstance],
    seed: u64,
) -> (Vec<SteerInstance>, Vec<SteerInstance>, f64) {
    const MEASURE_MS: u64 = 5000;
    let cfg = WorldConfig { seed, ..WorldConfig::for_con(&con) };
    let field = OdorField { x: 1.0e6, y: 1.0e6, strength: 0.0, radius_mm: 8.0 };
    let mut sim = Sim::new(Arc::clone(con), params.clone(), map, field, cfg);
    sim.apply_steer_instances(steer_insts);
    sim.add_steer_instances(vis_steer);
    sim.vis_objects = drum(0.0);

    for _ in 0..WARMUP_MS {
        sim.step();
    }
    // windowed mean of the arbiter EMA rates (sample every 100 ms)
    let mut sum: HashMap<u64, f64> = HashMap::new();
    let mut samples = 0u64;
    for t in 0..MEASURE_MS {
        sim.step();
        if t % 100 == 0 {
            for (body, hz) in sim.steer_instance_stats() {
                *sum.entry(body).or_default() += hz;
            }
            samples += 1;
        }
    }
    let mean = |b: u64| sum.get(&b).copied().unwrap_or(0.0) / samples as f64;

    let pre_bias: f64 = sim
        .steer_instance_stats()
        .iter()
        .filter_map(|&(b, r)| {
            steer_insts
                .iter()
                .chain(vis_steer)
                .find(|i| i.body_id == b)
                .map(|i| i.w * (r - i.base))
        })
        .sum();
    let center = |insts: &[SteerInstance]| -> Vec<SteerInstance> {
        insts
            .iter()
            .map(|i| SteerInstance { base: mean(i.body_id), ..*i })
            .collect()
    };
    let (c_odor, c_vis) = (center(steer_insts), center(vis_steer));
    let out = serde_json::to_vec_pretty(&json!({
        "seed": seed,
        "measure_ms": MEASURE_MS,
        "pre_centering_bias_hz": pre_bias,
        "odor": c_odor, "visual": c_vis,
    }))
    .unwrap();
    let path = format!(
        "{}-centered.json",
        std::env::args().nth(4).unwrap_or_else(|| "runs/optomotor".into())
    );
    fs::write(&path, out).unwrap();
    (c_odor, c_vis, pre_bias)
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
        ..WorldConfig::for_con(&con)
    };
    // the kernel RNG stream is keyed on params.seed — re-key per run
    let mut params = params.clone();
    params.seed = seed;
    // no odor in this assay: field parked far away at zero strength
    let field = OdorField { x: 1.0e6, y: 1.0e6, strength: 0.0, radius_mm: 8.0 };
    let mut sim = Sim::new(Arc::clone(con), params.clone(), map, field, cfg);
    sim.apply_steer_instances(steer_insts);
    sim.add_steer_instances(vis_steer);
    // warm up with a STATIC scene: the assay measures the response to
    // drum ONSET (the behavioral response adapts within seconds, so the
    // onset transient is the signal — a 30 s mean would average it away)
    sim.vis_objects = drum(0.0);

    for _ in 0..WARMUP_MS {
        sim.step();
    }
    sim.vis_objects = drum(drum_omega);

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
            "onset_window_s": seconds,
            "protocol": "static warmup, then drum onset; mean omega over window",
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
