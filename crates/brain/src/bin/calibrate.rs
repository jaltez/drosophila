use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use brain::{Brain, LifParams, FP_ONE};
use connectome::Connectome;

/// Per-superclass homeostatic calibration of LIF gains (2D: E and I).
///
/// Standing drive = sensory background + intrinsic background. Intrinsic
/// noise on all neurons removes the silent absorbing state, so rates grade
/// smoothly with gains. Per round: measure per-superclass rates, then move
/// each superclass's excitatory gain toward target and its inhibitory gain
/// the opposite way (rate too high -> exc down, inh up), both damped
/// (ratio^(1/4), clamped). Global threshold valves only for extremes.

const TARGET_HZ: f64 = 2.0;
const WARMUP_TICKS: usize = 500;
const MEASURE_TICKS: usize = 2500;
const RATIO_EXP: f64 = 0.25;

fn measure(con: &Arc<Connectome>, params: &LifParams) -> (Vec<u64>, f64) {
    let n_sup = con.superclass_names.len();
    let mut b = Brain::new(Arc::clone(con), params.clone());
    b.reseed_background(params.seed);
    for _ in 0..WARMUP_TICKS {
        b.tick(&[]);
    }
    let mut sup_spikes = vec![0u64; n_sup];
    let t0 = Instant::now();
    for _ in 0..MEASURE_TICKS {
        for &s in b.tick(&[]) {
            sup_spikes[con.superclass_codes[s as usize] as usize] += 1;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    (sup_spikes, MEASURE_TICKS as f64 / secs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("../../bridge/out/connectome-malecns-v1.0");
    let out_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("../../params/homeostasis_v1.json");
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(48);

    let con = Arc::new(Connectome::load(Path::new(con_dir)).expect("load connectome"));
    let n_sup = con.superclass_names.len();
    let mut per_sup_neurons = vec![0.0f64; n_sup];
    for &s in &con.superclass_codes {
        per_sup_neurons[s as usize] += 1.0;
    }
    let total_neurons: f64 = per_sup_neurons.iter().sum();
    let measure_secs = MEASURE_TICKS as f64 / 1000.0;

    let mut params = LifParams::default_for(&con);
    println!(
        "calibrating: {} neurons, {} superclasses, target {} Hz, max {} rounds",
        con.body_ids.len(),
        n_sup,
        TARGET_HZ,
        rounds
    );
    println!(
        "{:>4} {:>9} {:>8} {:>9} {:>8} {:>10}",
        "rnd", "mean Hz", "min Hz", "max Hz", "thresh", "ticks/s"
    );

    let mut stable_rounds = 0;
    for round in 0..rounds {
        let (sup_spikes, tps) = measure(&con, &params);
        let rates: Vec<f64> = (0..n_sup)
            .map(|s| sup_spikes[s] as f64 / per_sup_neurons[s] / measure_secs)
            .collect();
        let mean = sup_spikes.iter().sum::<u64>() as f64 / total_neurons / measure_secs;
        let mut min = f64::INFINITY;
        let mut max = 0.0f64;
        for &r in &rates {
            if r > 0.0 {
                min = min.min(r);
                max = max.max(r);
            }
        }

        println!(
            "{:>4} {:>9.3} {:>8.3} {:>9.3} {:>8} {:>10.0}",
            round,
            mean,
            if min.is_finite() { min } else { 0.0 },
            max,
            params.thresh,
            tps
        );

        if (1.0..=4.0).contains(&mean) && max < 60.0 {
            stable_rounds += 1;
            if stable_rounds >= 2 {
                println!("stable for 2 consecutive rounds; stopping early");
                break;
            }
        } else {
            stable_rounds = 0;
        }

        // damped 2D gain control: exc toward target, inh opposite
        for s in 0..n_sup {
            let ratio = (TARGET_HZ / rates[s].max(1e-9))
                .powf(RATIO_EXP)
                .clamp(0.9, 1.111);
            params.gains_exc[s] = ((params.gains_exc[s] as f64) * ratio)
                .clamp(0.05 * FP_ONE as f64, 20.0 * FP_ONE as f64) as i64;
            // inhibition never below unity (never weaken inhibition)
            params.gains_inh[s] = ((params.gains_inh[s] as f64) / ratio)
                .clamp(FP_ONE as f64, 40.0 * FP_ONE as f64) as i64;
        }
        // gentle global threshold valves for extremes only
        if mean < 0.05 {
            params.thresh = (params.thresh * 7 / 8).max(256);
        }
        if mean > 80.0 {
            params.thresh = (params.thresh * 9 / 8).min(512 * 400);
        }
    }

    // final verification run
    let (sup_spikes, tps) = measure(&con, &params);
    println!("\nfinal per-superclass rates (Hz):");
    for (s, name) in con.superclass_names.iter().enumerate() {
        let rate = sup_spikes[s] as f64 / per_sup_neurons[s] / measure_secs;
        println!(
            "  {:<24} {:>7.3}   exc {:.2}  inh {:.2}",
            name,
            rate,
            params.gains_exc[s] as f64 / FP_ONE as f64,
            params.gains_inh[s] as f64 / FP_ONE as f64
        );
    }

    if let Some(parent) = Path::new(out_path).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(out_path, serde_json::to_vec_pretty(&params).unwrap()).unwrap();
    println!("\nwrote {} ({:.0} ticks/s during measure)", out_path, tps);
}
