use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use brain::{Brain, LifParams};
use connectome::{Connectome, FUNC_DN, FUNC_MOTOR};

/// Sensory-storm experiment: baseline -> photoreceptor rate x10 -> recovery.
/// Writes tick-indexed spike log + summary JSON; verifies determinism by
/// replaying the first N ticks in a fresh Brain.

const BASELINE_TICKS: u32 = 2000;
const STIM_TICKS: u32 = 2000;
const POST_TICKS: u32 = 1000;
const STIM_MULT: f64 = 10.0;
const VERIFY_TICKS: u32 = 500;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("../../bridge/out/connectome-malecns-v1.0");
    let params_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("../../params/homeostasis_v1.json");
    let out_dir = args.get(3).map(|s| s.as_str()).unwrap_or("../../runs/stim-v1");
    let stim_class = args.get(4).map(|s| s.as_str()).unwrap_or("ol_sensory");

    let con = Arc::new(Connectome::load(Path::new(con_dir)).expect("load connectome"));
    let params: LifParams = serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    let n_sup = con.superclass_names.len();
    let ol = con
        .superclass_names
        .iter()
        .position(|n| n == stim_class)
        .expect("ol_sensory superclass");
    let base_rate = params.sensory_rates_hz[ol];

    let per_sup: Vec<f64> = con.superclass_codes.iter().fold(
        vec![0.0f64; n_sup],
        |mut c, &s| {
            c[s as usize] += 1.0;
            c
        },
    );
    let dn: Vec<u32> = con.indices_of_class(FUNC_DN);
    let mn: Vec<u32> = con.indices_of_class(FUNC_MOTOR);
    let is_dn = |i: u32| con.func_classes[i as usize] == FUNC_DN;
    let is_mn = |i: u32| con.func_classes[i as usize] == FUNC_MOTOR;

    // --- run ---
    let mut b = Brain::new(con.clone(), params.clone());
    b.reseed_background(params.seed);
    let mut log: Vec<(u32, u32)> = Vec::with_capacity(2_000_000);
    let t0 = Instant::now();
    let total = BASELINE_TICKS + STIM_TICKS + POST_TICKS;

    let mut first_dn_after_stim: Option<u32> = None;
    let mut first_mn_after_stim: Option<u32> = None;

    let mut sup_base = vec![0u64; n_sup];
    let mut sup_stim = vec![0u64; n_sup];
    let mut dn_base = 0u64;
    let mut dn_stim = 0u64;
    let mut mn_base = 0u64;
    let mut mn_stim = 0u64;

    for t in 0..total {
        if t == BASELINE_TICKS {
            b.params_mut().sensory_rates_hz[ol] = base_rate * STIM_MULT;
        }
        if t == BASELINE_TICKS + STIM_TICKS {
            b.params_mut().sensory_rates_hz[ol] = base_rate;
        }
        let spikes = b.tick(&[]).to_vec();
        for &s in &spikes {
            log.push((t, s));
            let sup = con.superclass_codes[s as usize] as usize;
            if t < BASELINE_TICKS {
                sup_base[sup] += 1;
                if is_dn(s) {
                    dn_base += 1;
                }
                if is_mn(s) {
                    mn_base += 1;
                }
            } else if t < BASELINE_TICKS + STIM_TICKS {
                sup_stim[sup] += 1;
                if is_dn(s) {
                    dn_stim += 1;
                }
                if is_mn(s) {
                    mn_stim += 1;
                }
                if first_dn_after_stim.is_none() && is_dn(s) {
                    first_dn_after_stim = Some(t - BASELINE_TICKS);
                }
                if first_mn_after_stim.is_none() && is_mn(s) {
                    first_mn_after_stim = Some(t - BASELINE_TICKS);
                }
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let ticks_per_s = total as f64 / secs;

    // --- determinism verification: replay first VERIFY_TICKS ticks ---
    let mut b2 = Brain::new(con.clone(), params.clone());
    b2.reseed_background(params.seed);
    let mut replay = Vec::new();
    for t in 0..VERIFY_TICKS {
        for &s in b2.tick(&[]) {
            replay.push((t, s));
        }
    }
    let head: Vec<(u32, u32)> = log.iter().take(replay.len()).copied().collect();
    let det_ok = head == replay;
    let mut det_bytes = Vec::with_capacity(replay.len() * 8);
    for (t, s) in &replay {
        det_bytes.extend_from_slice(&t.to_le_bytes());
        det_bytes.extend_from_slice(&s.to_le_bytes());
    }
    let det_hash = fnv1a(&det_bytes);

    // --- write outputs ---
    fs::create_dir_all(out_dir).unwrap();
    let mut bin = Vec::with_capacity(log.len() * 8);
    for (t, s) in &log {
        bin.extend_from_slice(&t.to_le_bytes());
        bin.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(Path::new(out_dir).join("spikes.bin"), &bin).unwrap();

    let rate = |c: u64, n: f64, ticks: u32| c as f64 / n / (ticks as f64 / 1000.0);
    let mut sup_rows = serde_json::Map::new();
    for s in 0..n_sup {
        sup_rows.insert(
            con.superclass_names[s].clone(),
            serde_json::json!({
                "n": per_sup[s],
                "baseline_hz": rate(sup_base[s], per_sup[s], BASELINE_TICKS),
                "stim_hz": rate(sup_stim[s], per_sup[s], STIM_TICKS),
            }),
        );
    }
    let summary = serde_json::json!({
        "connectome": { "neurons": con.body_ids.len(), "edges": con.csr_targets.len() },
        "params_seed": params.seed,
        "phases": {
            "baseline_ticks": BASELINE_TICKS,
            "stim_ticks": STIM_TICKS,
            "post_ticks": POST_TICKS,
            "stimulus": { "superclass": stim_class, "mult": STIM_MULT }
        },
        "dn": { "n": dn.len(), "baseline_hz": rate(dn_base, dn.len() as f64, BASELINE_TICKS),
                 "stim_hz": rate(dn_stim, dn.len() as f64, STIM_TICKS),
                 "first_spike_ms_after_onset": first_dn_after_stim },
        "motor": { "n": mn.len(), "baseline_hz": rate(mn_base, mn.len() as f64, BASELINE_TICKS),
                    "stim_hz": rate(mn_stim, mn.len() as f64, STIM_TICKS),
                    "first_spike_ms_after_onset": first_mn_after_stim },
        "per_superclass": sup_rows,
        "perf": { "ticks_per_s": ticks_per_s, "realtime_factor": ticks_per_s / 1000.0 },
        "determinism": { "verified": det_ok, "replay_ticks": VERIFY_TICKS,
                          "fnv1a_first_spikes": format!("{:016x}", det_hash) },
        "total_spikes": log.len(),
    });
    let summary_path = Path::new(out_dir).join("summary.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary).unwrap()).unwrap();

    println!("spikes logged: {}", log.len());
    println!("DN   baseline {:.2} Hz -> stim {:.2} Hz (first after onset: {:?} ms)",
        rate(dn_base, dn.len() as f64, BASELINE_TICKS),
        rate(dn_stim, dn.len() as f64, STIM_TICKS), first_dn_after_stim);
    println!("MN   baseline {:.2} Hz -> stim {:.2} Hz (first after onset: {:?} ms)",
        rate(mn_base, mn.len() as f64, BASELINE_TICKS),
        rate(mn_stim, mn.len() as f64, STIM_TICKS), first_mn_after_stim);
    println!("perf: {:.0} ticks/s ({:.2}x realtime)", ticks_per_s, ticks_per_s / 1000.0);
    println!("determinism: {} (fnv1a {:016x})", det_ok, det_hash);
    println!("wrote {}", out_dir);
}
