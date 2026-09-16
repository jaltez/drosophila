//! Loader for extracted connectome artifacts (see bridge/extract.py).
//!
//! Binary layout (all little-endian, arrays aligned to neurons sorted by bodyId):
//! - `body_ids.u64`      n × u64   MaleCNS bodyId per neuron
//! - `func_classes.u8`   n × u8    0=inter, 1=sensory, 2=dn, 3=motor
//! - `csr_offsets.u64`   (n+1) × u64   row offsets into edge arrays
//! - `csr_targets.u32`   m × u32   target neuron index
//! - `csr_weights.i16`   m × i16   signed fixed-point weight (SCALE=512)

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const WEIGHT_SCALE: f32 = 512.0;
pub const FUNC_INTER: u8 = 0;
pub const FUNC_SENSORY: u8 = 1;
pub const FUNC_DN: u8 = 2;
pub const FUNC_MOTOR: u8 = 3;


#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub counts: Counts,
}

#[derive(Debug, Deserialize)]
pub struct Counts {
    pub neurons: u64,
    pub edges: u64,
    pub synapses: u64,
}

#[derive(Debug)]
pub struct Connectome {
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub body_ids: Vec<u64>,
    /// 0=inter, 1=sensory, 2=dn, 3=motor (same order as body_ids)
    pub func_classes: Vec<u8>,
    /// Superclass index per neuron (into `superclass_names`)
    pub superclass_codes: Vec<u8>,
    pub superclass_names: Vec<String>,
    /// Soma side per neuron: 0 unknown, 1 left, 2 right, 3 midline.
    pub side_codes: Vec<u8>,
    /// Cell-type index per neuron (into `type_names`; 65535 = untyped).
    pub type_codes: Vec<u16>,
    pub type_names: Vec<String>,
    /// Entry-nerve index per neuron (into `entry_nerve_names`; 0 = none).
    pub entry_nerve_codes: Vec<u16>,
    pub entry_nerve_names: Vec<String>,
    /// Per-neuron position in mm, if the artifact carries positions.
    pub positions_mm: Option<Vec<[f32; 3]>>,
    /// Electrical (gap) synapses: (a, b, weight fp4.12), bidirectional.
    pub gap_edges: Vec<(u32, u32, i64)>,
    pub csr_offsets: Vec<u64>,
    pub csr_targets: Vec<u32>,
    pub csr_weights: Vec<i16>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("missing file {0}")]
    Missing(PathBuf),
    #[error("size mismatch {file}: expected {expected} bytes, got {actual}")]
    Size { file: PathBuf, expected: u64, actual: u64 },
    #[error("bad manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

fn read_exact<T: Copy>(path: &Path, expect: usize) -> Result<Vec<T>, LoadError> {
    let bytes = fs::read(path).map_err(|_| LoadError::Missing(path.into()))?;
    let size = std::mem::size_of::<T>();
    if bytes.len() != expect * size {
        return Err(LoadError::Size {
            file: path.into(),
            expected: (expect * size) as u64,
            actual: bytes.len() as u64,
        });
    }
    Ok(bytes
        .chunks_exact(size)
        .map(|c| unsafe { c.align_to::<T>().1[0] })
        .collect())
}

impl Connectome {
    pub fn load(dir: &Path) -> Result<Self, LoadError> {
        let manifest: Manifest =
            serde_json::from_str(&fs::read_to_string(dir.join("manifest.json"))?)?;
        let n = manifest.counts.neurons as usize;
        let m = manifest.counts.edges as usize;
        Ok(Self {
            dir: dir.into(),
            manifest,
            body_ids: read_exact(&dir.join("body_ids.u64"), n)?,
            func_classes: read_exact(&dir.join("func_classes.u8"), n)?,
            superclass_codes: read_exact(&dir.join("superclass_codes.u8"), n)?,
            superclass_names: serde_json::from_str(
                &fs::read_to_string(dir.join("superclass_names.json"))?)?,
            csr_offsets: read_exact(&dir.join("csr_offsets.u64"), n + 1)?,
            csr_targets: read_exact(&dir.join("csr_targets.u32"), m)?,
            csr_weights: read_exact(&dir.join("csr_weights.i16"), m)?,
            side_codes: read_exact(&dir.join("side_codes.u8"), n)?,
            type_codes: read_exact(&dir.join("type_codes.u16"), n)?,
            type_names: serde_json::from_str(
                &fs::read_to_string(dir.join("type_names.json"))?)?,
            entry_nerve_codes: read_exact(&dir.join("entry_nerve_codes.u16"), n)?,
            entry_nerve_names: serde_json::from_str(
                &fs::read_to_string(dir.join("entry_nerve_names.json"))?)?,
            gap_edges: Vec::new(),
            positions_mm: {
                let f = dir.join("positions.f32");
                if f.exists() {
                    let bytes = fs::read(&f)?;
                    if bytes.len() != n * 12 {
                        return Err(LoadError::Size {
                            file: f,
                            expected: (n * 12) as u64,
                            actual: bytes.len() as u64,
                        });
                    }
                    Some(
                        bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect::<Vec<f32>>()
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect(),
                    )
                } else {
                    None
                }
            }
        })
    }

    /// Indices of neurons in a functional class (0=inter,1=sensory,2=dn,3=motor).
    pub fn indices_of_class(&self, class: u8) -> Vec<u32> {
        (0..self.func_classes.len() as u32)
            .filter(|&i| self.func_classes[i as usize] == class)
            .collect()
    }

    /// Out-edges of neuron `i` as (target, edge_index, weight).
    pub fn out_edges_indexed(
        &self,
        i: u32,
    ) -> impl Iterator<Item = (u32, u64, i16)> + '_ {
        let (a, b) = (self.csr_offsets[i as usize], self.csr_offsets[i as usize + 1]);
        (a..b).map(move |e| {
            let e = e as usize;
            (self.csr_targets[e], e as u64, self.csr_weights[e])
        })
    }

    /// Out-edges of neuron `i` as (target, weight) pairs.
    pub fn out_edges(&self, i: u32) -> impl Iterator<Item = (u32, i16)> + '_ {
        let (a, b) = (self.csr_offsets[i as usize], self.csr_offsets[i as usize + 1]);
        (a..b).map(|e| {
            let e = e as usize;
            (self.csr_targets[e], self.csr_weights[e])
        })
    }

    /// Neuron index by bodyId, if present.
    pub fn index_of_body(&self, body_id: u64) -> Option<u32> {
        self.body_ids
            .binary_search(&body_id)
            .ok()
            .map(|i| i as u32)
    }

    /// Index of a cell type name, if present.
    pub fn type_index(&self, name: &str) -> Option<u16> {
        self.type_names.iter().position(|t| t == name).map(|i| i as u16)
    }

    /// All neuron indices of a given type (e.g. "DNg13").
    pub fn indices_of_type(&self, name: &str) -> Vec<u32> {
        match self.type_index(name) {
            Some(code) => (0..self.type_codes.len() as u32)
                .filter(|&i| self.type_codes[i as usize] == code)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Total signed fixed-point out-weight per neuron; positive input means
    /// net excitatory drive.
    pub fn net_out_weight(&self, i: u32) -> i64 {
        self.out_edges(i).map(|(_, w)| w as i64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the extracted artifact (built by bridge/extract.py).
    fn artifact() -> Option<PathBuf> {
        let p = Path::new("../../bridge/out/connectome-malecns-v1.0");
        p.is_dir().then(|| p.to_path_buf())
    }

    #[test]
    fn loads_artifact_and_consistency() {
        let Some(dir) = artifact() else { return }; // skipped if not extracted
        let c = Connectome::load(&dir).unwrap();
        let n = c.body_ids.len();
        assert_eq!(n, c.func_classes.len());
        assert_eq!(n + 1, c.csr_offsets.len());
        assert_eq!(c.csr_offsets[n], c.csr_targets.len() as u64);
        assert_eq!(c.csr_targets.len(), c.csr_weights.len());
        assert!(c.csr_offsets.windows(2).all(|w| w[0] <= w[1]));
        // every CSR row is sorted by target index (extraction sorts by (pre, post))
        for i in 0..n as u32 {
            let t: Vec<u32> = c.out_edges(i).map(|(t, _)| t).collect();
            assert!(t.windows(2).all(|w| w[0] < w[1]), "row {i} unsorted/dup");
            assert!(t.iter().all(|&t| t != i), "autapse at {i}");
        }
        let dn = c.indices_of_class(2);
        let motor = c.indices_of_class(3);
        let sensory = c.indices_of_class(1);
        eprintln!(
            "n={n} edges={} dn={} motor={} sensory={}",
            c.csr_targets.len(),
            dn.len(),
            motor.len(),
            sensory.len()
        );
    }
}
