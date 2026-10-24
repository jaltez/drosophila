# drosophila

Deterministic whole-brain simulation of adult *Drosophila melanogaster*, built on
real connectomes, with embodied behavioral assays.

> [!WARNING]
> **Archived research snapshot (October 2026).** This repository is published as a
> frozen record of a research project, not as a maintained tool. It is provided
> **as-is, with no support, no roadmap, and no planned updates**; issues and pull
> requests are disabled. The results below are research-grade and deliberately
> include documented failures — this is **not** a validated behavioral model of a
> fly (one assay arm passes at n=15, several others fail; see
> [Status at freeze](#status-at-freeze)). Nothing here is fit for production,
> clinical, or robotics use. The connectome artifacts are **not** included in the
> repository and must be rebuilt from public datasets (~15 GB) to run anything.
> Code and configuration are released under CC-BY-4.0; the underlying connectome
> datasets have their own licenses and citation requirements (see
> [Provenance](#provenance)).

## What this is

The project asks one question: **does brain wiring alone produce behavior?**

It loads a real, whole-CNS insect connectome — the male adult fly (*MaleCNS*
v1.0, 164,488 neurons / 25.3M edges / 122.9M synapses) and the female adult brain
(FlyWire *Fafb* v783, 138,590 neurons / 14.9M edges / 53.7M synapses) — into a
leaky integrate-and-fire spiking kernel at 1 ms ticks, attaches minimal sensory
transducers (two-antenna odor field, ommatidia, sugar contact) and a kinematic
body in a 40 mm-radius arena, and scores what the fly *does* against
odor-free controls. There is no task training anywhere: walk, escape, feeding,
grooming, casting and surging all come out of the wiring plus per-superclass
homeostatic gains.

Three engineering commitments define the codebase:

1. **Determinism as a contract.** Identical artifact + params + seed + sensory
   timeline ⇒ bit-identical spike sequence. The kernel is integer-only
   (i16 fixed-point weights, SCALE=512; i64 membrane), synaptic accumulation is
   integer-atomic so thread order cannot matter, and the RNG is a
   platform-independent xorshift64\*. Enforced by tests (FNV-1a spike-hash
   replay, bit-identical resume) in CI.
2. **Honesty.** Failed assays are committed as FAILs with root causes. Every
   weight-set flip and mechanism fix points at recorded evidence
   (e.g. `scenarios/visual_steer_calibration.json` records that the visual
   steering channel was anti-signed and documents the open-loop transfer used
   to fix it).
3. **Evidence-gated curation.** Neuron→behavior assignments come from literature
   anchors (Namiki et al. 2018 DN taxonomy; Bidaye et al. 2014/2015 MDN; giant
   fiber DNp01), the project's own simulated DN screens, and an AI decision-model
   seeding pipeline (TypeSafe/Jev) whose 1,762 cached answers are committed
   (`bridge/out/tsafe_cache.json`) so curation is reproducible offline. Seeds
   below 0.9 confidence are held for review, never auto-promoted.

Performance: ~800–1,300 ticks/s on a laptop CPU for the full male CNS
(~0.8–1.3× realtime at 1 ms ticks), no GPU.

## Status at freeze

Results of the final battery round (n=15 seeds per arm, 30 s episodes, paired
exact sign-flip permutation tests vs odor-free control):

| Assay | Arm | Result |
|---|---|---|
| Chemotaxis lock-on, cast-surge controller | male | **PASS** on source-hold: Δ **+1.42 s**, p = 0.031 (`runs/lockon3-male`). Both arms approach the source; approach distance Δ −0.78 mm, p = 0.19 (ns). |
| Chemotaxis lock-on | female | **FAIL**: approach Δ −0.69 mm, p = 0.78; hold/plume p = 1.0 (`runs/lockon3-female`). Root cause: the female homeostasis (descending gains ×5) co-boosts the MDN backward vote to 5.1 s/run, fighting the surges. Needs a calibration-evidence round, not metric tuning. |
| DN instance-level steering decode (continuous) | — | **Unusable at the walk plant**: +4 Hz bearing signal against 100–300 Hz per-instance differential noise (~30× SNR deficit). Replaced by the cast-surge macro-controller for chemotaxis; decode still drives optomotor/visual assays. Measurement to beat recorded in `docs/kernel-realism-notes.md`. |
| Optomotor (12-stripe drum, onset protocol) | — | **Mixed per seed**: the visual channel was found anti-signed, flipped with recorded provenance, but the 9-seed aggregate does not yet separate from static control. |
| DN odor-screen sex conservation | both | 450 shared DN types; male/female pooled response correlation **r = +0.766** (odor +0.810, sugar +0.872); 24 types lateralized in both sexes. |
| Functional dimorphism | both | FDI mean 0.38; **fruDsx** enrichment **z = +2.46** (anatomical dimorphism z = +1.83) (`runs/functional_dimorphism.json`). |
| Leaf-level DN discrimination | — | **Negative result**: forward-walk (DNa01/02) vs backward (MDN) DNs do not separate in output anatomy (axis ratio 0.86 < 2 required; escape outgroup DNp01 does separate). Behavior-map promotion policy unchanged (`bridge/runs/leaf_screen.json`). |

## Architecture

Rust workspace (`crates/`, edition 2021) plus a Python bridge for data
extraction and analysis.

| Crate | Role |
|---|---|
| `connectome` | Loads binary artifacts: CSR edge arrays with signed i16 fixed-point weights (SCALE=512), per-neuron function class / superclass / side / entry-nerve codes, soma positions, gap-junction scaffold, `manifest.json` with SHA-256 provenance of inputs and outputs. |
| `brain` | LIF kernel at 1 ms ticks: per-superclass E/I gains with in-norm cap, spike-frequency adaptation, per-edge conduction delays from soma distance (0.5 mm/ms, clamped [1, 20] ticks), lazy Tsodyks–Markram depression, optional gap junctions (off — no source data). Bins: `calibrate` (homeostatic gain solver, 2 Hz class target), `dn_screen`, `dn_battery`, `gain_probe`, `stim` (sensory-storm determinism replay). |
| `transduce` | Sensory transducers (point-source odor field sampled per antenna at ±0.6 mm, two-eye ommatidia, sugar contact) and the motor decode: `DnArbiter` EMA's descending-neuron rates into walk/escape/feed/groom votes (`scenarios/behavior_map.json`) and ~130 instance-level signed steering weights (`scenarios/steer_instances*.json`) with a 1 s DC high-pass. |
| `world` | Engine-independent closed loop (brain + arbiter + body + arena); one `Sim::step` = 1 ms. Contains the cast-surge plume-tracking controller, escape burst trigger, contact-gated feeding. Bins: `sim_headless`, `optomotor`, `sweep` (grid runner + paired permutation stats), `steer_tf`, `fork_experiment` (odor-removal counterfactual + resume identity proof), `fly_window` (Bevy viewer, behind the `render` feature). |
| `runlog` | Run manifests (JSON) and tick-indexed Parquet traces sampled at 100 ms. |

`bridge/` (Python 3 + numpy/pandas/pyarrow) extracts both connectomes into the
artifact format, constructs steering weights from measured DN screens, runs the
AI curation pipeline, and produces the statistical analyses. Shared vocabulary
in `bridge/vocab.py` is "loud-unmapped": unknown annotation values are reported
and fail the run under `--strict`.

## Reproducing

```bash
# 1. Build (Rust stable)
cargo build --release

# 2. Data — public sources, ~15 GB, placed in bridge/data/
#    MaleCNS v1.0 (FlyEM/Janelia):
gsutil -m cp "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/*.feather" bridge/data/
#    FlyWire Fafb v783 (female):
curl -L -o bridge/data/proofread_connections_783.feather \
  https://zenodo.org/records/10676866/files/proofread_connections_783.feather
curl -L -o bridge/data/proofread_root_ids_783.npy \
  https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy
curl -L -o bridge/data/flywire_annotations.tsv \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv

python3 -m venv bridge/.venv && bridge/.venv/bin/pip install numpy pandas pyarrow matplotlib

# 3. Extract connectome artifacts (bridge/out/connectome-*/; ~245 MB, gitignored)
cd bridge && .venv/bin/python extract.py && .venv/bin/python extract_flywire.py && cd ..

# 4. Calibrate homeostasis (writes params/homeostasis_v1.json)
cargo run --release -p brain --bin calibrate -- bridge/out/connectome-malecns-v1.0 params/homeostasis_v1.json

# 5. Assays
cargo run --release -p world --bin sim_headless --            # chemotaxis episode (positional args, sensible defaults)
cargo run --release -p world --bin sweep -- \
  bridge/out/connectome-malecns-v1.0 params/homeostasis_v1.json \
  scenarios/behavior_map.json grids/lockon3-male.json         # n-seed battery + permutation stats
cargo run --release -p world --bin optomotor -- --help        # open-loop drum assay
cargo run --release -p world --bin fly_window --features render  # interactive viewer (Bevy)

# 6. Tests (determinism, resume identity, steering sign; CI-clean without artifacts)
cargo test --workspace
cargo test --release -p brain determinism_synthetic
python3 bridge/test_vocab.py && python3 bridge/test_steer_balance.py
```

Notes:

- Step 2–3 are only needed once; the extractors pin SHA-256 hashes of every
  input and output in `manifest.json`, so a rebuilt artifact is verifiable.
- Grid definitions used for the frozen results are in `grids/*.json` (lockon3
  male/female, cast-surge tuning calib-cs1..3, sweep-tau*).
- `runs/` (74 MB of traces and summaries, gitignored) was regenerated locally;
  `sweep_summary.json` files referenced above are the statistical verdicts.

## Repository layout

```
crates/          Rust workspace (connectome, brain, transduce, world, runlog)
bridge/          Python: dataset extraction, weight construction, curation, analysis
  out/           gitignored artifacts + committed tsafe_cache.json
  data/          gitignored source datasets (~15 GB)
scenarios/       behavior map, steering instance sets, visual calibration (tracked)
params/          homeostasis solutions (tracked)
grids/           sweep/battery grid definitions (tracked)
docs/            kernel realism notes and parked decisions
runs/            gitignored simulation outputs
```

## Known limitations / parked

- Gap junctions: kernel support exists but MaleCNS v1.0 and FlyWire v783 ship
  no electrical-coupling table; off by default, parked 2026-09-29.
- Conduction delays use straight-line soma distance; skeleton-path refinement
  parked (no result shows delay-order sensitivity).
- The body is kinematic (2D arena, fixed speeds); no physics, no legs. For a
  full neuromechanical body see the NeuroMechFly/flygym line of work.
- Continuous steering from DN instance decodes is not solvable at this kernel's
  SNR (see status table); chemotaxis uses the cast-surge macro-controller.

## Provenance

Connectome data (not redistributed here):

- **MaleCNS v1.0** — male adult fly CNS connectome, FlyEM team / Janelia
  (HHMI), `gs://flyem-male-cns/v1.0/`.
- **FlyWire Fafb v783** — female adult brain proofread release,
  Dorkenwald et al., Zenodo record
  [10676866](https://zenodo.org/records/10676866); annotations from the
  [flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
  repository.

Behavior anchors: Namiki et al. 2018 (descending neuron taxonomy), Bidaye et al.
2014/2015 (MDN backward walking), giant-fiber escape (DNp01).

## License

Code, configuration and documentation: **CC-BY-4.0** (see `LICENSE` and
`Cargo.toml`). The upstream connectome datasets carry their own terms and
citation requirements; using them through this pipeline does not change that.
No warranty of any kind, express or implied.
