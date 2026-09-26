//! Dump the log-mel for the golden fixture's signal, for offline comparison.
//!
//! Not a test: it writes a binary file so the front end can be diffed against
//! the reference with numpy, which is far faster to iterate in than a Rust
//! test with no visualisation.
//!
//!   cargo run -p vaani-asr --example dump_logmel -- <fixture> <preproc dir> <out.bin>

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: dump_logmel <fixture.json> <artefacts dir> <out.bin>");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&a[1]).expect("fixture");
    let v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    let sig: Vec<f32> = v["signal"]
        .as_array()
        .expect("signal")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect();
    let dir = std::path::PathBuf::from(&a[2]);
    let p = vaani_asr::FrontendParams::load(&dir).expect("preproc");
    let t = vaani_asr::FrontendTables::load(&dir, &p).expect("tables");
    let m = vaani_asr::log_mel(&sig, &p, &t).expect("log_mel");
    let mut bytes = Vec::with_capacity(m.data.len() * 4);
    for v in &m.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&a[3], &bytes).expect("write");
    eprintln!(
        "wrote {} values ({} mels x {} frames, {} valid) to {}",
        m.data.len(),
        m.n_mels,
        m.n_frames,
        m.valid_frames,
        a[3]
    );
}
