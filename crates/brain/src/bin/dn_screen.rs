//! DN screen: which descending-neuron types respond differentially to
//! unilateral olfactory drive? Produces steer-vote candidates for the
//! behavior map from circuit routing rather than literature guesses.
//!
//! Three runs (baseline / left-antenna / right-antenna odor gain), per-type
//! DN firing rates, ranked by L-R asymmetry. Writes dn_screen.json.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use brain::{Brain, LifParams};
use connectome::Connectome;

const WARMUP_MS: u64 = 2000;
const MEASURE_MS: u64 = 6000;
const ODOR_GAIN: f32 = 8.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args.get(1).map(|s| s.as_str()).unwrap_or("bridge/out/connectome-malecns-v1.0");
    let params_path = args.get(2).map(|s| s.as_str()).unwrap_or("params/homeostasis_v1.json");
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("runs/dn_screen.json");

    let con = Connectome::load(Path::new(con_dir)).expect("load connectome");
    let params: LifParams = serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    let cb_sensory = con
        .superclass_names
        .iter()
        .position(|n| n == "cb_sensory")
        .expect("cb_sensory") as u8;

    // sided cb_sensory neurons
    let mut orns_l = Vec::new();
    let mut orns_r = Vec::new();
    for i in 0..con.func_classes.len() as u32 {
        let j = i as usize;
        if con.func_classes[j] == connectome::FUNC_SENSORY
            && con.superclass_codes[j] == cb_sensory
        {
            match con.side_codes[j] {
                1 => orns_l.push(i),
                2 => orns_r.push(i),
                _ => {}
            }
        }
    }
    println!(
        "cb_sensory: {} left, {} right ORNs; odor gain x{ODOR_GAIN}",
        orns_l.len(),
        orns_r.len()
    );

    // DN bookkeeping (per-instance, not aggregated by type)
    let dn: Vec<u32> = (0..con.func_classes.len() as u32)
        .filter(|&i| con.func_classes[i as usize] == connectome::FUNC_DN)
        .collect();

    let con_arc = Arc::new(con);
    let run_inst = |stim_side: u8| -> HashMap<u32, f64> {
        let mut b = Brain::new(Arc::clone(&con_arc), params.clone());
        b.reseed_background(params.seed + 1000 + stim_side as u64);
        let gains = b.sensory_gains_mut();
        for &i in &orns_l {
            gains[i as usize] = if stim_side == 1 { ODOR_GAIN } else { 1.0 };
        }
        for &i in &orns_r {
            gains[i as usize] = if stim_side == 2 { ODOR_GAIN } else { 1.0 };
        }
        for _ in 0..WARMUP_MS {
            b.tick(&[]);
        }
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for _ in 0..MEASURE_MS {
            for &s in b.tick(&[]) {
                *counts.entry(s).or_insert(0) += 1;
            }
        }
        let secs = MEASURE_MS as f64 / 1000.0;
        dn.iter()
            .map(|&i| {
                let c = counts.get(&i).copied().unwrap_or(0) as f64;
                (i, c / secs)
            })
            .collect()
    };

    println!("running baseline / left / right ...");
    let baseline = run_inst(0);
    let left = run_inst(1);
    let right = run_inst(2);

    // per-instance asymmetry: rate under left drive minus right drive
    let mut rows: Vec<(f64, String, u8, u64, f64, f64, f64)> = dn
        .iter()
        .map(|&i| {
            let r0 = baseline.get(&i).copied().unwrap_or(0.0);
            let rl = left.get(&i).copied().unwrap_or(0.0);
            let rr = right.get(&i).copied().unwrap_or(0.0);
            let asym = (rl - rr) / (rl + rr + 1e-9);
            let strength = (rl + rr) / 2.0;
            let score = asym * strength.max(0.05);
            (
                score,
                con_arc.type_names
                    .get(con_arc.type_codes[i as usize] as usize)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "untyped".into()),
                con_arc.side_codes[i as usize],
                con_arc.body_ids[i as usize],
                rl,
                rr,
                r0,
            )
        })
        .collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!(
        "{:<10} {:>4} {:>10} {:>8} {:>8} {:>7}",
        "type", "side", "bodyId", "L Hz", "R Hz", "score"
    );
    for (score, name, side, body, rl, rr, r0) in rows.iter().take(24) {
        println!(
            "{:<10} {:>4} {:>10} {:>8.3} {:>8.3} {:>7.3}  base {:.2}",
            name, side, body, rl, rr, score, r0
        );
    }

    let json = rows
        .iter()
        .map(|(score, name, side, body, rl, rr, r0)| {
            serde_json::json!({
                "type": name, "side": side, "bodyId": body,
                "left_hz": rl, "right_hz": rr, "base_hz": r0, "score": score,
            })
        })
        .collect::<Vec<_>>();
    if let Some(parent) = Path::new(out_path).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(out_path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    println!("wrote {}", out_path);
}
