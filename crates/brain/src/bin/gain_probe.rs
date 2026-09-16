//! Gain probe: multiply one superclass's gains by a factor, run the
//! calibrated brain, report named-type base rates + the class mean.
//! Used to repair per-type silence under class-mean calibration.
//!
//! usage: gain_probe <con_dir> <params> <superclass> <mult> [type names...]

use std::fs;
use std::path::Path;
use std::sync::Arc;

use brain::{Brain, LifParams};
use connectome::Connectome;

const WARMUP_MS: u64 = 4000;
const MEASURE_MS: u64 = 8000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args.get(1).map(|s| s.as_str()).unwrap_or("bridge/out/connectome-flywire-v783");
    let params_path = args.get(2).map(|s| s.as_str()).unwrap_or("params/homeostasis_female_v1.json");
    let sup_name = args.get(3).map(|s| s.as_str()).unwrap_or("descending");
    let mult: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let types: Vec<String> = args.iter().skip(5).cloned().collect();

    let con = Arc::new(Connectome::load(Path::new(con_dir)).expect("load connectome"));
    let mut params: LifParams = serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    let sup = con
        .superclass_names
        .iter()
        .position(|n| n == sup_name)
        .expect("superclass") as u8;

    if mult != 1.0 {
        params.gains_exc[sup as usize] = (params.gains_exc[sup as usize] as f64 * mult) as i64;
        params.gains_inh[sup as usize] = (params.gains_inh[sup as usize] as f64 * mult) as i64;
    }

    // members of the class and of the probed types
    let members: Vec<u32> = (0..con.func_classes.len() as u32)
        .filter(|&i| con.superclass_codes[i as usize] == sup)
        .collect();
    let mut probes: Vec<(String, Vec<u32>)> = Vec::new();
    for t in &types {
        if let Some(k) = con.type_index(t) {
            probes.push((
                t.clone(),
                (0..con.func_classes.len() as u32)
                    .filter(|&i| con.type_codes[i as usize] == k)
                    .collect(),
            ));
        } else {
            eprintln!("type {t} not in artifact");
        }
    }

    let mut b = Brain::new(Arc::clone(&con), params.clone());
    b.reseed_background(params.seed + 42);
    for _ in 0..WARMUP_MS {
        b.tick(&[]);
    }
    let mut counts: HashMap<u32, u64> = Default::default();
    let mut class_total = 0u64;
    for _ in 0..MEASURE_MS {
        for s in b.tick(&[]) {
            class_total += 1;
            *counts.entry(*s).or_insert(0) += 1;
        }
    }
    let secs = MEASURE_MS as f64 / 1000.0;
    let class_mean = class_total as f64 / secs / members.len() as f64;
    println!(
        "mult {mult} class {sup_name}: mean {class_mean:.2} Hz over {} neurons",
        members.len()
    );
    for (t, idx) in &probes {
        let rates: Vec<f64> = idx
            .iter()
            .map(|&i| counts.get(&i).copied().unwrap_or(0) as f64 / secs)
            .collect();
        let mean = rates.iter().sum::<f64>() / rates.len().max(1) as f64;
        println!("  {t}: mean {mean:.2} Hz per-instance {rates:?}");
    }
}

use std::collections::HashMap;
