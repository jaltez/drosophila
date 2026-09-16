//! Grid runner: runs a config × seed sweep of the headless chemotaxis sim.
//!
//! Usage: sweep <con_dir> <params_path> <map_path> <grid_json>
//!
//! Grid JSON:
//! {
//!   "configs": [{"name": "cfgA", "odor_k": 12.0, "dc_tau_ms": 1000.0,
//!                "surge_gain": 1.5, "steer_gain_rad_s": 8.0,
//!                "odor": [25.0, 0.0], "control": false}],
//!   "seeds": [7, 11],
//!   "seconds": 40,
//!   "out_root": "runs/grid"
//! }
//!
//! All override fields are optional; omitted ones keep the WorldConfig /
//! BehaviorMap defaults. For each config × seed the sim loop runs inline
//! (connectome loaded once), writing runs/grid/<config>/s<seed>/trace.parquet
//! + summary.json. One aggregate sweep_summary.json holds per-run stats and a
//! markdown table; control configs (odor parked far away) provide the baseline.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use brain::LifParams;
use connectome::Connectome;
use serde::{Deserialize, Serialize};
use serde_json::json;
use transduce::{BehaviorMapFile, OdorField, SteerInstance};
use world::{Sim, WorldConfig};

fn default_odor() -> [f64; 2] {
    [25.0, 0.0]
}

fn default_seconds() -> u64 {
    40
}

fn default_out_root() -> String {
    "runs/grid".to_string()
}

#[derive(Debug, Deserialize, Serialize)]
struct GridConfig {
    name: String,
    odor_k: Option<f64>,
    dc_tau_ms: Option<f64>,
    surge_gain: Option<f64>,
    steer_gain_rad_s: Option<f64>,
    #[serde(default = "default_odor")]
    odor: [f64; 2],
    #[serde(default)]
    control: bool,
}

#[derive(Debug, Deserialize)]
struct GridDef {
    configs: Vec<GridConfig>,
    seeds: Vec<u64>,
    #[serde(default = "default_seconds")]
    seconds: u64,
    #[serde(default = "default_out_root")]
    out_root: String,
}

struct RunStats {
    config: String,
    control: bool,
    seed: u64,
    out_dir: PathBuf,
    d_start: f64,
    d_end: f64,
    d_min: f64,
    d_mean: f64,
    hold_ms: u64,
    plume_ms: u64,
    labels_ms: HashMap<&'static str, u64>,
    ticks_per_s: f64,
}

fn sim_dist(sim: &Sim, field: &OdorField) -> f64 {
    let (dx, dy) = (sim.fly.x - field.x, sim.fly.y - field.y);
    (dx * dx + dy * dy).sqrt()
}

fn run_one(
    con: &Arc<Connectome>,
    base_params: &LifParams,
    map_bytes: &[u8],
    steer_insts: &[SteerInstance],
    vis_steer: &[SteerInstance],
    gc: &GridConfig,
    seed: u64,
    seconds: u64,
    out_dir: &Path,
) -> RunStats {
    let mut params = base_params.clone();
    params.seed = seed;

    // Fresh map parse per run = the map copy the dc_tau override lands in.
    let mut map: BehaviorMapFile = serde_json::from_slice(map_bytes).expect("parse behavior map");
    if let Some(tau) = gc.dc_tau_ms {
        map.arbiter.steer_dc_tau_ms = tau;
    }

    let mut cfg = WorldConfig::for_con(con);
    cfg.seed = seed;
    if let Some(k) = gc.odor_k {
        cfg.odor_k = k;
    }
    if let Some(g) = gc.surge_gain {
        cfg.surge_gain = g;
    }
    if let Some(s) = gc.steer_gain_rad_s {
        cfg.steer_gain_rad_s = s;
    }

    let field = OdorField {
        x: gc.odor[0],
        y: gc.odor[1],
        strength: 1.0,
        radius_mm: 8.0,
    };

    let mut sim = Sim::new(Arc::clone(con), params, &map, field, cfg.clone());
    sim.apply_steer_instances(steer_insts);
    sim.add_steer_instances(vis_steer);

    // In-assay centering (lock-on instrument): null each instance's base
    // against its measured no-odor rate in this arena, so steering carries
    // directional modulation rather than construction bias.
    let mut pre_bias = 0.0;
    if std::env::var("CHEMO_CENTER").as_deref() != Ok("0") {
        pre_bias = sim.center_steer_bases(2000, 5000);
    }

    // warm up: 2 s before recording (EMA settling)
    for _ in 0..2000 {
        sim.step();
    }

    let total_ticks = seconds * 1000;
    let sample_every = 100; // ms
    let mut trace =
        Vec::with_capacity((total_ticks / sample_every + 1) as usize);
    let mut labels: HashMap<&'static str, u64> = Default::default();
    let d0 = sim_dist(&sim, &field);
    let mut d_min = f64::INFINITY;
    let mut d_sum = 0.0;
    let mut hold_ms = 0u64;
    let mut plume_ms = 0u64;
    let t0 = Instant::now();
    for t in 0..total_ticks {
        let info = sim.step();
        *labels.entry(info.label).or_insert(0) += 1;
        let d = sim_dist(&sim, &field);
        d_sum += d;
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
            trace.push(runlog::TraceRecord {
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
    let d_mean = d_sum / total_ticks as f64;
    let ticks_per_s = total_ticks as f64 / wall;

    fs::create_dir_all(out_dir).unwrap();
    runlog::write_trace_parquet(&out_dir.join("trace.parquet"), &trace).unwrap();
    let summary = json!({
        "config": {
            "name": gc.name,
            "control": gc.control,
            "seconds": seconds, "seed": seed,
            "overrides": {
                "odor_k": gc.odor_k,
                "dc_tau_ms": gc.dc_tau_ms,
                "surge_gain": gc.surge_gain,
                "steer_gain_rad_s": gc.steer_gain_rad_s,
            },
            "odor": {"x": field.x, "y": field.y, "radius_mm": field.radius_mm,
                     "k": cfg.odor_k},
            "effective": {"surge_gain": cfg.surge_gain,
                          "steer_gain_rad_s": cfg.steer_gain_rad_s,
                          "dc_tau_ms": map.arbiter.steer_dc_tau_ms},
            "arena_radius_mm": cfg.arena_radius_mm,
            "connectome": {"neurons": con.body_ids.len(),
                            "edges": con.csr_targets.len()},
        },
        "dist_to_source": {
            "start_mm": d0,
            "end_mm": d_end,
            "min_mm": d_min,
            "mean_mm": d_mean,
            "approached": d_end < d0 - 1.0,
        },
        "source_hold_ms": hold_ms,
        "plume_ms": plume_ms,
        "pre_centering_bias_hz": pre_bias,
        "labels_ms": labels,
        "perf": {
            "ticks": total_ticks,
            "ticks_per_s": ticks_per_s,
            "realtime_factor": total_ticks as f64 / wall / 1000.0,
        },
        "trace_first": trace.first(),
        "trace_last": trace.last(),
    });
    runlog::RunManifest::write(&out_dir.join("summary.json"), &summary).unwrap();

    RunStats {
        config: gc.name.clone(),
        control: gc.control,
        seed,
        out_dir: out_dir.to_path_buf(),
        d_start: d0,
        d_end,
        d_min,
        d_mean,
        hold_ms,
        plume_ms,
        labels_ms: labels,
        ticks_per_s,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: sweep <con_dir> <params_path> <map_path> <grid_json>");
        std::process::exit(2);
    }
    let con_dir = PathBuf::from(&args[1]);
    let params_path = PathBuf::from(&args[2]);
    let map_path = PathBuf::from(&args[3]);
    let grid_path = PathBuf::from(&args[4]);

    let grid: GridDef =
        serde_json::from_slice(&fs::read(&grid_path).unwrap()).expect("parse grid json");
    let con = Arc::new(Connectome::load(&con_dir).expect("load connectome"));
    let mut base_params: LifParams =
        serde_json::from_slice(&fs::read(&params_path).unwrap()).unwrap();
    if base_params.gains_exc.len() != con.superclass_names.len() {
        base_params = LifParams::default_for(&con);
    }
    let map_bytes = fs::read(&map_path).unwrap();
    let steer_insts: Vec<SteerInstance> =
        serde_json::from_slice(&fs::read("scenarios/steer_instances.json").unwrap())
            .expect("load steer instances");
    let vis_steer: Vec<SteerInstance> = fs::read("scenarios/visual_steer_instances.json")
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();

    let out_root = PathBuf::from(&grid.out_root);
    fs::create_dir_all(&out_root).unwrap();

    let mut stats = Vec::new();
    for gc in &grid.configs {
        for seed in &grid.seeds {
            let run_dir = out_root.join(&gc.name).join(format!("s{seed}"));
            let st = run_one(
                &con, &base_params, &map_bytes, &steer_insts, &vis_steer, gc,
                *seed, grid.seconds, &run_dir,
            );
            println!(
                "[{}/s{}] dist {} -> {} mm (min {:.1}, mean {:.1})",
                gc.name, seed, st.d_start, st.d_end, st.d_min, st.d_mean
            );
            stats.push(st);
        }
    }

    // Control baseline: mean of per-run mean distances across control configs.
    let control_runs: Vec<&RunStats> = stats.iter().filter(|s| s.control).collect();
    let control_mean = if control_runs.is_empty() {
        None
    } else {
        Some(control_runs.iter().map(|s| s.d_mean).sum::<f64>() / control_runs.len() as f64)
    };

    // Markdown table: config | per-seed mean dist | mean over seeds | vs control.
    let mut header: Vec<String> = vec!["config".to_string()];
    for seed in &grid.seeds {
        header.push(format!("mean d s{seed} (mm)"));
    }
    header.push("mean over seeds (mm)".to_string());
    if control_mean.is_some() {
        header.push("Δ vs control (mm)".to_string());
    }
    let mut table = format!(
        "| {} |\n|{}|\n",
        header.join(" | "),
        header.iter().map(|_| "---").collect::<Vec<_>>().join("|")
    );
    for gc in &grid.configs {
        let runs: Vec<&RunStats> =
            stats.iter().filter(|s| s.config == gc.name).collect();
        let mean: f64 = runs.iter().map(|s| s.d_mean).sum::<f64>() / runs.len() as f64;
        let mut row = vec![gc.name.clone()];
        for seed in &grid.seeds {
            match runs.iter().find(|s| s.seed == *seed) {
                Some(s) => row.push(format!("{:.2}", s.d_mean)),
                None => row.push("—".to_string()),
            }
        }
        row.push(format!("{:.2}", mean));
        if let Some(cm) = control_mean {
            row.push(format!("{:+.2}", mean - cm));
        }
        table += &format!("| {} |\n", row.join(" | "));
    }

    let sweep_summary = json!({
        "grid": {
            "configs": &grid.configs,
            "seeds": &grid.seeds,
            "seconds": grid.seconds,
            "out_root": &grid.out_root,
        },
        "control_mean_dist_mm": control_mean,
        "runs": stats.iter().map(|s| json!({
            "config": s.config,
            "control": s.control,
            "seed": s.seed,
            "out_dir": s.out_dir.to_string_lossy(),
            "dist_to_source": {
                "start_mm": s.d_start,
                "end_mm": s.d_end,
                "min_mm": s.d_min,
                "mean_mm": s.d_mean,
            },
            "source_hold_ms": s.hold_ms,
            "plume_ms": s.plume_ms,
            "labels_ms": s.labels_ms,
            "ticks_per_s": s.ticks_per_s,
        })).collect::<Vec<_>>(),
        "table_md": table.trim_end(),
    });
    let summary_path = out_root.join("sweep_summary.json");
    runlog::RunManifest::write(&summary_path, &sweep_summary).unwrap();

    print!("\n{}", table);
    println!("control baseline (mean dist): {}", match control_mean {
        Some(cm) => format!("{:.2} mm", cm),
        None => "n/a (no control config)".to_string(),
    });
    println!("wrote {}", summary_path.display());
}
