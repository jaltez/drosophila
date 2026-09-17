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

/// Headless chemotaxis runner arguments. Positional form is kept for
/// compatibility with the historical call signature.
#[derive(clap::Parser)]
struct Args {
    #[arg(default_value = "bridge/out/connectome-malecns-v1.0")]
    con_dir: String,
    #[arg(default_value = "params/homeostasis_v1.json")]
    params_path: String,
    #[arg(default_value = "scenarios/behavior_map.json")]
    map_path: String,
    #[arg(default_value = "runs/world-v1")]
    out_dir: String,
    #[arg(default_value_t = 60)]
    seconds: u64,
    #[arg(default_value_t = 7)]
    seed: u64,
    #[arg(default_value_t = 25.0)]
    odor_x: f64,
    #[arg(default_value_t = 0.0)]
    odor_y: f64,
    #[arg(default_value_t = 12.0)]
    odor_k: f64,
    #[arg(default_value_t = 0.0)]
    obj_x: f64,
    #[arg(default_value_t = 0.0)]
    obj_y: f64,
    #[arg(default_value_t = 0.0)]
    obj_brightness: f64,
    /// NaN (the default) means no sugar source
    #[arg(default_value_t = f64::NAN)]
    sugar_x: f64,
    #[arg(default_value_t = f64::NAN)]
    sugar_y: f64,
    #[arg(default_value = "scenarios/steer_instances.json")]
    inst_file: String,
}

fn main() {
    let a = <Args as clap::Parser>::parse();
    let Args {
        con_dir, params_path, map_path, out_dir, seconds, seed,
        odor_x, odor_y, odor_k, obj_x, obj_y, obj_brightness,
        sugar_x, sugar_y, inst_file,
    } = a;
    let sugar = if sugar_x.is_nan() { None } else { Some((sugar_x, sugar_y)) };

    let con = Connectome::load(Path::new(&con_dir)).expect("load connectome");
    let mut params: LifParams =
        serde_json::from_slice(&fs::read(&params_path).unwrap()).unwrap();
    params.seed = seed;
    if params.gains_exc.len() != con.superclass_names.len() {
        params = LifParams::default_for(&con);
    }
    let map: BehaviorMapFile =
        serde_json::from_slice(&fs::read(&map_path).unwrap()).unwrap();
    let steer_insts: Vec<transduce::SteerInstance> =
        serde_json::from_slice(&fs::read(&inst_file).unwrap()).unwrap();
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

    // In-assay centering (lock-on instrument): null each steering
    // instance's base against its measured no-odor rate in this arena.
    let pre_bias = if std::env::var("CHEMO_CENTER").as_deref() != Ok("0") {
        sim.center_steer_bases(2000, 5000)
    } else {
        0.0
    };

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
    let mut hold_ms = 0u64;
    let mut plume_ms = 0u64;
    let t0 = Instant::now();
    for t in 0..total_ticks {
        let info = sim.step();
        *labels.entry(info.label).or_insert(0) += 1;
        let d = sim_dist(&sim, &field);
        if d <= 5.0 {
            hold_ms += 1;
        }
        if d <= 12.0 {
            plume_ms += 1;
        }
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

    fs::create_dir_all(&out_dir).unwrap();
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
        "source_hold_ms": hold_ms,
        "plume_ms": plume_ms,
        "pre_centering_bias_hz": pre_bias,
        "labels_ms": labels,
        "perf": {
            "ticks": total_ticks,
            "ticks_per_s": total_ticks as f64 / wall,
            "realtime_factor": total_ticks as f64 / wall / 1000.0,
        },
        "trace_first": trace.first(),
        "trace_last": trace.last(),
    });
    runlog::write_trace_parquet(&Path::new(&out_dir).join("trace.parquet"), &trace)
        .unwrap();
    fs::write(
        Path::new(&out_dir).join("summary.json"),
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
