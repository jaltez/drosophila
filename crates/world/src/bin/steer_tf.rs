//! Static steer transfer function: fly held at origin heading 0, odor
//! source swept through bearings at fixed distance; measure steady-state
//! steer_drive. Fresh brain per bearing (no adaptation carryover).

use std::fs;
use std::path::Path;
use std::sync::Arc;

use brain::LifParams;
use connectome::Connectome;
use serde_json::json;
use transduce::{BehaviorMapFile, OdorField};
use world::{Sim, WorldConfig};

const WARMUP_MS: u64 = 2000;
const MEASURE_MS: u64 = 3000;
const SOURCE_DIST_MM: f64 = 20.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args.get(1).map(|s| s.as_str()).unwrap_or("bridge/out/connectome-malecns-v1.0");
    let params_path = args.get(2).map(|s| s.as_str()).unwrap_or("params/homeostasis_v1.json");
    let map_path = args.get(3).map(|s| s.as_str()).unwrap_or("scenarios/behavior_map.json");
    let out_path = args.get(4).map(|s| s.as_str()).unwrap_or("runs/steer_tf.json");

    let con = Connectome::load(Path::new(con_dir)).expect("load connectome");
    let mut params: LifParams = serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    if params.gains_exc.len() != con.superclass_names.len() {
        params = LifParams::default_for(&con);
    }
    let map: BehaviorMapFile = serde_json::from_slice(&fs::read(map_path).unwrap()).unwrap();
    let inst_file = args.get(5).map(|s| s.as_str()).unwrap_or("scenarios/steer_instances.json");
    let steer_insts: Vec<transduce::SteerInstance> =
        serde_json::from_slice(&fs::read(inst_file).unwrap()).unwrap();
    let cfg = WorldConfig::for_con(&con);

    let con_arc = Arc::new(con);
    let mut rows = Vec::new();
    println!("{:>8} {:>10} {:>10}", "bearing", "steer Hz", "walk Hz");
    for deg in [-150i32, -120, -90, -60, -30, -15, 15, 30, 60, 90, 120, 150] {
        let e = deg as f64 / 180.0 * std::f64::consts::PI;
        let field = OdorField {
            x: SOURCE_DIST_MM * e.cos(),
            y: SOURCE_DIST_MM * e.sin(),
            strength: 1.0,
            radius_mm: 8.0,
        };
        let mut sim = Sim::new(
            Arc::clone(&con_arc),
            params.clone(),
            &map,
            field,
            cfg.clone(),
        );
        sim.apply_steer_instances(&steer_insts);
        let mut acc_steer = 0.0f64;
        let mut acc_walk = 0.0f64;
        for t in 0..(WARMUP_MS + MEASURE_MS) {
            // hold the fly: restore pose each tick (sim still ticks the brain)
            sim.fly.x = 0.0;
            sim.fly.y = 0.0;
            sim.fly.heading = 0.0;
            let info = sim.step();
            if t >= WARMUP_MS {
                acc_steer += info.drives.steer;
                acc_walk += info.drives.walk;
            }
        }
        let n = MEASURE_MS as f64;
        let steer = acc_steer / n;
        let walk = acc_walk / n;
        println!("{:>8} {:>10.2} {:>10.2}", deg, steer, walk);
        rows.push(json!({"bearing_deg": deg, "steer_hz": steer, "walk_hz": walk}));
    }
    fs::write(out_path, serde_json::to_vec_pretty(&json!(rows)).unwrap()).unwrap();
    println!("wrote {}", out_path);
}
