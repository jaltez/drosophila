//! Fork-at-tick counterfactual: run to T with an odor source, checkpoint,
//! then replay three ways:
//!   A: continue unchanged (baseline future)
//!   B: continue with the odor REMOVED (counterfactual)
//!   C: restore checkpoint and continue unchanged — must equal A
//!      bit-for-bit (resume determinism proof in the wild).

use std::fs;
use std::path::Path;
use std::sync::Arc;

use brain::LifParams;
use connectome::Connectome;
use serde_json::json;
use transduce::{BehaviorMapFile, OdorField};
use world::{Sim, WorldConfig};

fn run_tail(sim: &mut Sim, ticks: u64) -> Vec<(u64, f64, f64)> {
    let mut trace = Vec::new();
    for _ in 0..ticks {
        let info = sim.step();
        trace.push((info.t_ms, info.fly.x, info.fly.y));
    }
    trace
}

fn mean_dist_to_source(sim: &Sim, trace: &[(u64, f64, f64)]) -> f64 {
    let (sx, sy) = (sim.field.x, sim.field.y);
    trace
        .iter()
        .map(|&(_, x, y)| ((x - sx).powi(2) + (y - sy).powi(2)).sqrt())
        .sum::<f64>()
        / trace.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args.get(1).map(|s| s.as_str()).unwrap_or("bridge/out/connectome-malecns-v1.0");
    let params_path = args.get(2).map(|s| s.as_str()).unwrap_or("params/homeostasis_v1.json");
    let map_path = args.get(3).map(|s| s.as_str()).unwrap_or("scenarios/behavior_map.json");
    let out_dir = args.get(4).map(|s| s.as_str()).unwrap_or("runs/fork-v1");
    let pre_ticks: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let post_ticks: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(7);

    let con = Connectome::load(Path::new(con_dir)).expect("load connectome");
    let params: LifParams = serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    let map: BehaviorMapFile = serde_json::from_slice(&fs::read(map_path).unwrap()).unwrap();
    let field = OdorField { x: 25.0, y: 0.0, strength: 1.0, radius_mm: 8.0 };
    let cfg = WorldConfig { seed, ..WorldConfig::for_con(&con) };

    let con_arc = Arc::new(con);
    let mut sim = Sim::new(Arc::clone(&con_arc), params.clone(), &map, field, cfg.clone());
    if let Ok(insts) = serde_json::from_slice::<Vec<transduce::SteerInstance>>(
        &fs::read("scenarios/steer_instances.json").unwrap_or_default(),
    ) {
        sim.apply_steer_instances(&insts);
    }
    if let Ok(vis) = serde_json::from_slice::<Vec<transduce::SteerInstance>>(
        &fs::read("scenarios/visual_steer_instances.json").unwrap_or_default(),
    ) {
        sim.add_steer_instances(&vis);
    }

    // warmup + pre-fork run (odor present throughout)
    for _ in 0..2000 {
        sim.step();
    }
    let pre = run_tail(&mut sim, pre_ticks);
    let checkpoint = sim.checkpoint();

    // branch A: unchanged future
    let a_tail = run_tail(&mut sim, post_ticks);
    let a_mean = mean_dist_to_source(&sim, &a_tail);

    // branch C: restore same checkpoint, unchanged continuation
    let mut sim_c = Sim::new(Arc::clone(&con_arc), params.clone(), &map, field, cfg.clone());
    if let Ok(insts) = serde_json::from_slice::<Vec<transduce::SteerInstance>>(
        &fs::read("scenarios/steer_instances.json").unwrap_or_default(),
    ) {
        sim_c.apply_steer_instances(&insts);
    }
    if let Ok(vis) = serde_json::from_slice::<Vec<transduce::SteerInstance>>(
        &fs::read("scenarios/visual_steer_instances.json").unwrap_or_default(),
    ) {
        sim_c.add_steer_instances(&vis);
    }
    sim_c.restore(&checkpoint);
    let c_tail = run_tail(&mut sim_c, post_ticks);
    let resume_identical = a_tail == c_tail;

    // branch B: counterfactual — odor removed at the fork point
    let mut sim_b = Sim::new(Arc::clone(&con_arc), params.clone(), &map, field, cfg);
    if let Ok(insts) = serde_json::from_slice::<Vec<transduce::SteerInstance>>(
        &fs::read("scenarios/steer_instances.json").unwrap_or_default(),
    ) {
        sim_b.apply_steer_instances(&insts);
    }
    if let Ok(vis) = serde_json::from_slice::<Vec<transduce::SteerInstance>>(
        &fs::read("scenarios/visual_steer_instances.json").unwrap_or_default(),
    ) {
        sim_b.add_steer_instances(&vis);
    }
    sim_b.restore(&checkpoint);
    sim_b.field = OdorField { x: -1000.0, y: -1000.0, strength: 0.0, radius_mm: 8.0 };
    let b_tail = run_tail(&mut sim_b, post_ticks);
    let b_mean = mean_dist_to_source(&sim_b, &b_tail);

    // divergence tick between A and B
    let div = a_tail
        .iter()
        .zip(&b_tail)
        .position(|(a, b)| a != b)
        .map(|p| p as u64);

    fs::create_dir_all(out_dir).unwrap();
    let wr = |name: &str, rows: &[(u64, f64, f64)]| {
        let t: Vec<_> = rows.iter().map(|(t, x, y)| json!({"t": t, "x": x, "y": y})).collect();
        fs::write(Path::new(out_dir).join(name), serde_json::to_vec(&t).unwrap()).unwrap();
    };
    wr("branch_a.json", &a_tail);
    wr("branch_b.json", &b_tail);
    wr("branch_c.json", &c_tail);
    fs::write(
        Path::new(out_dir).join("summary.json"),
        serde_json::to_vec_pretty(&json!({
            "fork_tick": pre_ticks, "post_ticks": post_ticks, "seed": seed,
            "resume_identical": resume_identical,
            "ab_divergence_offset_ms": div,
            "mean_dist_after_fork": {"A_odor": a_mean, "B_no_odor": b_mean},
            "pre_mean_dist": mean_dist_to_source(&sim, &pre),
        }))
        .unwrap(),
    )
    .unwrap();

    println!(
        "resume identical: {} | A/B divergence: {:?} ms after fork",
        resume_identical, div
    );
    println!(
        "mean dist after fork: A(odor) {:.1} mm vs B(no odor) {:.1} mm",
        a_mean, b_mean
    );
    println!("wrote {}", out_dir);
}
