//! DN stimulus battery: identical protocol battery on any calibrated
//! artifact, per-DN-instance mean rates per protocol. Protocols:
//! base / odor-L / odor-R (sided ORN gain), vis-L / vis-R (sided visual
//! superclass gain), sugar (MxLbN afferent gain). Output: one JSON array
//! of {type, side, bodyId, hz per protocol} rows — the input to
//! `bridge/dn_profile_battery.py`.
//!
//! usage: dn_battery <con_dir> <params> <out.json> [odor_class] [vis_class]
//! defaults: odor_class=cb_sensory (MaleCNS), vis_class=visual_projection.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use brain::{Brain, LifParams};
use connectome::Connectome;

const WARMUP_MS: u64 = 2000;
const MEASURE_MS: u64 = 6000;
const STIM_GAIN: f32 = 8.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con_dir = args.get(1).map(|s| s.as_str()).unwrap_or("bridge/out/connectome-malecns-v1.0");
    let params_path = args.get(2).map(|s| s.as_str()).unwrap_or("params/homeostasis_v1.json");
    let out_path = args.get(3).map(|s| s.as_str()).unwrap_or("runs/dn_battery.json");
    let odor_class = args.get(4).map(|s| s.as_str()).unwrap_or("cb_sensory");
    let vis_class = args.get(5).map(|s| s.as_str()).unwrap_or("visual_projection");

    let con = Arc::new(Connectome::load(Path::new(con_dir)).expect("load connectome"));
    let params: LifParams = serde_json::from_slice(&fs::read(params_path).unwrap()).unwrap();
    let sup_idx = |name: &str| {
        con.superclass_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("superclass {} not in artifact", name)) as u8
    };
    let odor_sup = sup_idx(odor_class);
    let vis_sup = sup_idx(vis_class);
    let mxbn = con
        .entry_nerve_names
        .iter()
        .position(|n| n == "MxLbN")
        .expect("MxLbN nerve") as u16;

    // stimulus neuron sets (side: 0=none, 1=L, 2=R)
    let mut odor = Vec::new();
    let mut vis = Vec::new();
    let mut sugar = Vec::new();
    for i in 0..con.func_classes.len() as u32 {
        let j = i as usize;
        let side = con.side_codes[j];
        if con.func_classes[j] == connectome::FUNC_SENSORY && con.superclass_codes[j] == odor_sup
        {
            odor.push((i, side));
        }
        if con.superclass_codes[j] == vis_sup {
            vis.push((i, side));
        }
        if con.entry_nerve_codes[j] == mxbn {
            sugar.push((i, side));
        }
    }
    let counts = |set: &[(u32, u8)]| {
        (set.len(), set.iter().filter(|(_, s)| *s == 1).count(), set.iter().filter(|(_, s)| *s == 2).count())
    };
    let (no, nl, nr) = counts(&odor);
    let (nv, vl, vr) = counts(&vis);
    println!(
        "{oc}: odor {no} ({nl}L/{nr}R), vis {vc}: {nv} ({vl}L/{vr}R), sugar {ns}",
        oc = odor_class, no = no, nl = nl, nr = nr,
        vc = vis_class, nv = nv, vl = vl, vr = vr, ns = sugar.len()
    );


    let dn: Vec<u32> = (0..con.func_classes.len() as u32)
        .filter(|&i| con.func_classes[i as usize] == connectome::FUNC_DN)
        .collect();
    println!("DN instances: {}", dn.len());

    // Protocol drive channels:
    // - odor/sugar are sensory afferents: kernel-native gain modulation
    // - visual neurons are interneuron-classified in MaleCNS (gains act
    //   only on FUNC_SENSORY background streams), so visual drive is an
    //   external spike train at ~16 Hz/neuron, desynchronized per neuron
    //   (deterministic stride schedule) — identical in both sexes.
    const EXT_PERIOD: u64 = 63; // ticks between spikes per neuron (~16 Hz)
    let protocols: Vec<(&str, Vec<(u32, f32)>, Vec<u32>)> = vec![
        ("base", Vec::new(), Vec::new()),
        (
            "odor_L",
            odor.iter().filter(|(_, s)| *s == 1).map(|&(i, _)| (i, STIM_GAIN)).collect(),
            Vec::new(),
        ),
        (
            "odor_R",
            odor.iter().filter(|(_, s)| *s == 2).map(|&(i, _)| (i, STIM_GAIN)).collect(),
            Vec::new(),
        ),
        (
            "vis_L",
            Vec::new(),
            vis.iter().filter(|(_, s)| *s == 1).map(|&(i, _)| i).collect(),
        ),
        (
            "vis_R",
            Vec::new(),
            vis.iter().filter(|(_, s)| *s == 2).map(|&(i, _)| i).collect(),
        ),
        ("sugar", sugar.iter().map(|&(i, _)| (i, STIM_GAIN)).collect(), Vec::new()),
    ];

    let names = Arc::clone(&con);
    let mut rows: Vec<serde_json::Value> = Vec::new();
    // run all protocols; each gets its own seeded Brain (paired across
    // protocols via the same background stream family)
    let mut results: HashMap<&str, HashMap<u32, f64>> = HashMap::new();
    for (pid, stim, ext_set) in &protocols {
        let mut b = Brain::new(Arc::clone(&con), params.clone());
        b.reseed_background(params.seed + 1000);
        {
            let gains = b.sensory_gains_mut();
            for &(i, g) in stim {
                gains[i as usize] = g;
            }
        }
        let ext_drive = |t: u64| -> Vec<u32> {
            ext_set
                .iter()
                .filter(|&&i| (t + i as u64 * 7919) % EXT_PERIOD == 0)
                .copied()
                .collect()
        };
        for t in 0..WARMUP_MS {
            b.tick(&ext_drive(t));
        }
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for tt in 0..MEASURE_MS {
            for &s in b.tick(&ext_drive(WARMUP_MS + tt)) {
                *counts.entry(s).or_insert(0) += 1;
            }
        }
        let secs = MEASURE_MS as f64 / 1000.0;
        let mut per: HashMap<u32, f64> = HashMap::new();
        for &i in &dn {
            let c = counts.get(&i).copied().unwrap_or(0) as f64;
            per.insert(i, c / secs);
        }
        results.insert(pid, per);
        println!("protocol {} done", pid);
    }

    for &i in &dn {
        let j = i as usize;
        let mut row = serde_json::json!({
            "type": names.type_names
                .get(names.type_codes[j] as usize)
                .cloned()
                .unwrap_or_else(|| "untyped".into()),
            "side": names.side_codes[j],
            "bodyId": names.body_ids[j],
        });
        for (pid, _, _) in &protocols {
            row[pid] = serde_json::json!(results.get(pid).unwrap().get(&i).copied().unwrap_or(0.0));
        }
        rows.push(row);
    }
    fs::write(out_path, serde_json::to_vec(&json_protected(&rows)).unwrap()).unwrap();
    println!("wrote {} ({} rows)", out_path, rows.len());
}

// plain array output (the analysis script reads a bare list)
fn json_protected(rows: &[serde_json::Value]) -> serde_json::Value {
    serde_json::Value::Array(rows.to_vec())
}
