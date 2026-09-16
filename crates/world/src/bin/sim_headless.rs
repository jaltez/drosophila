//! Headless chemotaxis runner: brain-driven fly in an odor arena, no
//! rendering. Writes trace.parquet (100 ms sampling) + summary.json.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use brain::LifParams;
use connectome::Connectome;
use serde_json::json;
use transduce::{BehaviorMapFile, OdorField};
use runlog::TraceRecord;
use world::{Sim, WorldConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args.get(1).map(|s| s.as_str()).unwrap_or("bridge/out/connectome-malecns-v1.0");
    let params_path = args.get(2).map(|s| s.as_str()).unwrap_or("params/homeostasis_v1.json");
    let map_path = args.get(3).map(|s| s.as_str()).unwrap_or("scenarios/behavior_map.json");
    let out_dir = args.get(4).map(|s| s.as_str()).unwrap_or("runs/world-v1");
    let seconds: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(60);
    let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(7);
    let odor_x: f64 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(25.0);
    let odor_y: f64 = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let odor_k: f64 = args.get(9).and_then(|s| s.parse().ok()).unwrap_or(12.0);
    let obj_x: f64 = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let obj_y: f64 = args.get(11).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let obj_brightness: f64 = args.get(12).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let sugar_x: f64 = args.get(13).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
    let sugar_y: f64 = args.get(14).and_then(|s| s.parse().ok()).unwrap_or(f64::NAN);
    let sugar = if sugar_x.is_nan() { None } else { Some((sugar_x, sugar_y)) };
    let inst_file = args.get(15).map(|s| s.as_str()).unwrap_or("scenarios/steer_instances.json");

    let con = Connectome::load(Path::new(con_dir)).expect("load connectome");
    let mut params: LifParams =
        serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    params.seed = seed;
    if params.gains_exc.len() != con.superclass_names.len() {
        params = LifParams::default_for(&con);
    }
    let map: BehaviorMapFile =
        serde_json::from_slice(&fs::read(map_path).unwrap()).unwrap();
    let steer_insts: Vec<transduce::SteerInstance> =
        serde_json::from_slice(&fs::read(inst_file).unwrap()).unwrap();
    let vis_steer: Vec<transduce::SteerInstance> = fs::read(
        "scenarios/visual_steer_instances.json",
    )
    .ok()
    .and_then(|b| serde_json::from_slice(&b).ok())
    .unwrap_or_default();
    let cfg = WorldConfig {
        seed,
        odor_k,
        ..WorldConfig::for_con(&con)
    };
    let field = OdorField {
        x: odor_x,
        y: odor_y,
        strength: 1.0,
        radius_mm: 8.0,
    };
    let vis_objects = if obj_brightness > 0.0 {
        vec![transduce::VisObject {
            x: obj_x,
            y: obj_y,
            radius_mm: 3.0,
            brightness: obj_brightness,
            pivot: (0.0, 0.0),
            ang_vel_rad_s: 0.0,
            ang0: 0.0,
        }]
    } else {
        Vec::new()
    };

    let sim_con = Arc::new(con);
    let con_for_summary = Arc::clone(&sim_con);
    let mut sim = Sim::new(sim_con, params, &map, field, cfg.clone());
    sim.apply_steer_instances(&steer_insts);
    sim.add_steer_instances(&vis_steer);
    sim.vis_objects = vis_objects;
    sim.sugar = sugar;
    // warm up: 2 s before recording (EMA settling)
    for _ in 0..2000 {
        sim.step();
    }

    let total_ticks = seconds * 1000;
    let sample_every = 100; // ms
    let mut trace = Vec::with_capacity((total_ticks / sample_every + 1) as usize);
    let mut labels: std::collections::HashMap<&'static str, u64> = Default::default();
    let d0 = sim_dist(&sim, &field);
    let mut d_min = f64::INFINITY;
    let t0 = Instant::now();
    for t in 0..total_ticks {
        let info = sim.step();
        *labels.entry(info.label).or_insert(0) += 1;
        let d = sim_dist(&sim, &field);
        if d < d_min {
            d_min = d;
        }
        if t % sample_every == 0 {
            trace.push(TraceRecord {
                t_ms: info.t_ms,
                x: info.fly.x,
                y: info.fly.y,
                heading: info.fly.heading,
                label: info.label.to_string(),
                v: info.v_mm_s,
                w_rad_s: info.w_rad_s,
                walk_hz: info.drives.walk,
                steer_hz: info.drives.steer,
                escape_hz: info.drives.escape,
            });
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let d_end = sim_dist(&sim, &field);

    fs::create_dir_all(out_dir).unwrap();
    let summary = json!({
        "config": {
            "seconds": seconds, "seed": seed,
            "odor": {"x": field.x, "y": field.y, "radius_mm": field.radius_mm, "k": odor_k},
            "visual_object": {"x": obj_x, "y": obj_y, "brightness": obj_brightness},
            "sugar": sugar.map(|(x, y)| json!({"x": x, "y": y})),
            "arena_radius_mm": cfg.arena_radius_mm,
            "connectome": {"neurons": con_for_summary.body_ids.len(),
                            "edges": con_for_summary.csr_targets.len()},
        },
        "dist_to_source": {
            "start_mm": d0,
            "end_mm": d_end,
            "min_mm": d_min,
            "approached": d_end < d0 - 1.0,
        },
        "labels_ms": labels,
        "perf": {
            "ticks": total_ticks,
            "ticks_per_s": total_ticks as f64 / wall,
            "realtime_factor": total_ticks as f64 / wall / 1000.0,
        },
        "trace_first": trace.first(),
        "trace_last": trace.last(),
    });
    runlog::write_trace_parquet(&Path::new(out_dir).join("trace.parquet"), &trace)
        .unwrap();
    fs::write(
        Path::new(out_dir).join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();

    println!(
        "dist to source: start {:.1} -> end {:.1} mm (min {:.1}), approached: {}",
        d0, d_end, d_min, d_end < d0 - 1.0
    );
    println!("labels (ms): {:?}", labels);
    println!(
        "perf: {:.0} ticks/s ({:.2}x realtime)",
        total_ticks as f64 / wall,
        total_ticks as f64 / wall / 1000.0
    );
    println!("wrote {}", out_dir);
}

fn sim_dist(sim: &Sim, field: &OdorField) -> f64 {
    let (dx, dy) = (sim.fly.x - field.x, sim.fly.y - field.y);
    (dx * dx + dy * dy).sqrt()
}
