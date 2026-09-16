//! Run manifests (JSON) and tick-indexed trace logs (Parquet).

use std::fs;
use std::path::Path;

use arrow::array::{Array, Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum RunlogError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// One sampled trace row (100 ms sampling in the standard runners).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    pub t_ms: u64,
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub label: String,
    pub v: f64,
    pub w_rad_s: f64,
    pub walk_hz: f64,
    pub steer_hz: f64,
    pub escape_hz: f64,
}

impl TraceRecord {
    pub fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("t_ms", DataType::UInt64, false),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("heading", DataType::Float64, false),
            Field::new("label", DataType::Utf8, false),
            Field::new("v", DataType::Float64, false),
            Field::new("w_rad_s", DataType::Float64, false),
            Field::new("walk_hz", DataType::Float64, false),
            Field::new("steer_hz", DataType::Float64, false),
            Field::new("escape_hz", DataType::Float64, false),
        ]))
    }

    fn to_batch(records: &[TraceRecord]) -> Result<RecordBatch, RunlogError> {
        let schema = Self::schema();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from_iter(records.iter().map(|r| r.t_ms))),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.x))),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.y))),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.heading))),
                Arc::new(StringArray::from_iter(
                    records.iter().map(|r| Some(r.label.as_str())),
                )),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.v))),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.w_rad_s))),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.walk_hz))),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.steer_hz))),
                Arc::new(Float64Array::from_iter(records.iter().map(|r| r.escape_hz))),
            ],
        )?;
        Ok(batch)
    }

    /// Decode all rows from a parquet file written by `write_trace_parquet`.
    pub fn read_trace_parquet(path: &Path) -> Result<Vec<TraceRecord>, RunlogError> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let mut out = Vec::new();
        for batch in reader {
            let batch = batch?;
            let t = batch
                .column_by_name("t_ms")
                .expect("t_ms column")
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let num = batch.num_rows();
            let get = |name: &str| -> Vec<f64> {
                batch
                    .column_by_name(name)
                    .expect("column")
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect()
            };
            let xs = get("x");
            let ys = get("y");
            let headings = get("heading");
            let vs = get("v");
            let ws = get("w_rad_s");
            let walks = get("walk_hz");
            let steers = get("steer_hz");
            let escapes = get("escape_hz");
            let labels = batch
                .column_by_name("label")
                .expect("label column")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..num {
                out.push(TraceRecord {
                    t_ms: t.value(i),
                    x: xs[i],
                    y: ys[i],
                    heading: headings[i],
                    label: labels.value(i).to_string(),
                    v: vs[i],
                    w_rad_s: ws[i],
                    walk_hz: walks[i],
                    steer_hz: steers[i],
                    escape_hz: escapes[i],
                });
            }
        }
        Ok(out)
    }
}

/// Write trace records as a Parquet file (one row per record).
pub fn write_trace_parquet(path: &Path, records: &[TraceRecord]) -> Result<(), RunlogError> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir)?;
        }
    }
    let batch = TraceRecord::to_batch(records)?;
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, TraceRecord::schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Run manifest: a JSON document describing one run (config, git-ish provenance,
/// summary pointers). Thin wrapper over serde_json.
pub struct RunManifest;

impl RunManifest {
    pub fn write(path: &Path, value: &serde_json::Value) -> Result<(), RunlogError> {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir)?;
            }
        }
        let file = fs::File::create(path)?;
        serde_json::to_writer_pretty(file, value)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: u64) -> TraceRecord {
        TraceRecord {
            t_ms: n * 100,
            x: n as f64 * 0.5,
            y: -(n as f64) * 1.25,
            heading: n as f64 * 0.1,
            label: if n % 2 == 0 { "walk" } else { "steer" }.to_string(),
            v: n as f64 * 0.7,
            w_rad_s: -0.25 * n as f64,
            walk_hz: 1.0 + n as f64 * 0.1,
            steer_hz: 0.5,
            escape_hz: 0.0,
        }
    }

    #[test]
    fn parquet_roundtrip() {
        let records: Vec<TraceRecord> = (0..5).map(sample).collect();
        let path = std::env::temp_dir().join(format!(
            "runlog_test_{}_trace.parquet",
            std::process::id()
        ));
        write_trace_parquet(&path, &records).expect("write parquet");
        let back = TraceRecord::read_trace_parquet(&path).expect("read parquet");
        fs::remove_file(&path).ok();
        assert_eq!(back.len(), 5);
        for (a, b) in records.iter().zip(back.iter()) {
            assert_eq!(a.t_ms, b.t_ms);
            assert_eq!(a.label, b.label);
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.heading, b.heading);
            assert_eq!(a.v, b.v);
            assert_eq!(a.w_rad_s, b.w_rad_s);
            assert_eq!(a.walk_hz, b.walk_hz);
            assert_eq!(a.steer_hz, b.steer_hz);
            assert_eq!(a.escape_hz, b.escape_hz);
        }
    }

    #[test]
    fn manifest_roundtrip() {
        let path = std::env::temp_dir().join(format!(
            "runlog_test_{}_manifest.json",
            std::process::id()
        ));
        let v = serde_json::json!({"name": "cfgA", "seed": 7, "mean_dist": 12.5});
        RunManifest::write(&path, &v).expect("write manifest");
        let back: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(back, v);
    }
}
