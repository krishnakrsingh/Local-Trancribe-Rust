//! Measurements, printed with `--nocapture` and asserted against loose bounds.
//!
//! Three of these are the numbers the brief calls "measured values, not
//! guesses", and one of them is the reason this file exists rather than a
//! comment: **discovery must be fast**, and "fast" is only true if somebody
//! measures it on a tree that looks like a real model directory.
//!
//! Run it as:
//!
//! ```text
//! cargo test -p vaani-models --release --test bench -- --nocapture --ignored
//! ```
//!
//! `--ignored`, because these print numbers rather than assert behaviour, and a
//! benchmark that fails the build on a loaded machine trains people to ignore
//! it. The *correctness* assertions that must always run live in
//! `discovery.rs` and `verification.rs`.

mod support;

use std::path::Path;
use std::time::{Duration, Instant};

use support::TempDir;
use vaani_models::manifest::Manifest;
use vaani_models::paths::ModelRoots;
use vaani_models::verify::Verifier;
use vaani_models::{CancelToken, ModelManager, discover};

fn best_of<F: FnMut() -> Duration>(runs: u32, mut f: F) -> Duration {
    let mut worst = Duration::MAX;
    for _ in 0..runs {
        let t = f();
        worst = worst.min(t);
    }
    worst
}

fn mib_per_s(bytes: u64, d: Duration) -> f64 {
    bytes as f64 / d.as_secs_f64() / (1024.0 * 1024.0)
}

fn sravaani_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/sravaani")
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[test]
#[ignore = "prints measurements; run with --nocapture --ignored"]
fn manifest_load_time() {
    let text = vaani_models::manifest::EMBEDDED_MANIFEST;
    let runs = 200;
    let best = best_of(runs, || {
        let t = Instant::now();
        for _ in 0..10 {
            let m = Manifest::parse(text).expect("valid");
            std::hint::black_box(m);
        }
        t.elapsed() / 10
    });
    println!(
        "BENCH manifest: {} bytes parsed in {best:?} ({:.1} us/parse)",
        text.len(),
        best.as_secs_f64() * 1e6
    );
    assert!(
        best < Duration::from_millis(5),
        "manifest load regressed: {best:?}"
    );

    // And a manifest with N models, because the catalog grows.
    for n in [1usize, 10, 100, 1000] {
        let mut models = Vec::with_capacity(n);
        for i in 0..n {
            models.push(format!(
                r#"{{"id":"m{i}","engine_id":"e{i}","version":"1","display_name":"M{i}",
                "description":"d","task":"asr","architecture":"a","access":"open",
                "license":{{"spdx":"MIT","name":"MIT","url":"https://example.org/l"}},
                "source":{{"kind":"bundled"}},"languages":["hi","en"],
                "sample_rates":[16000],"size_bytes":4,
                "files":[{{"path":"w{i}.bin","role":"weights","bytes":4,
                  "sha256":"{sha}"}}]}}"#,
                sha = "0".repeat(64)
            ));
        }
        let big = format!(r#"{{"schema_version":1,"models":[{}]}}"#, models.join(","));
        let best = best_of(50, || {
            let t = Instant::now();
            for _ in 0..5 {
                std::hint::black_box(Manifest::parse(&big).expect("valid"));
            }
            t.elapsed() / 5
        });
        println!(
            "BENCH manifest: {n} model(s), {} KiB parsed in {best:?} ({:.1} us/parse)",
            big.len() / 1024,
            best.as_secs_f64() * 1e6
        );
        assert!(
            best < Duration::from_millis(250),
            "a {n}-model catalog took {best:?}, which would show as a startup pause"
        );
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

#[test]
#[ignore = "prints measurements; run with --nocapture --ignored"]
fn discovery_time_on_the_real_artefacts_tree() {
    let dir = sravaani_dir();
    let Some(root) = dir.parent().map(Path::to_path_buf) else {
        println!("skipping: artifacts/sravaani has no parent");
        return;
    };
    if !dir.join("encoder.fp16.onnx").is_file() {
        println!("skipping: {} is not present", dir.display());
        return;
    }
    let m = Manifest::embedded().expect("valid");
    let roots = ModelRoots::at(root.join("__absent__"), &root);
    let _ = discover::discover(&m, &roots);

    let d = best_of(200, || {
        let t = Instant::now();
        let r = discover::discover(&m, &roots);
        std::hint::black_box(&r);
        t.elapsed()
    });
    println!("BENCH discovery: the real 2.5 GiB artifacts tree (10 files, 2.7 GB) in {d:?}");
    assert!(d < Duration::from_millis(50), "discovery regressed: {d:?}");

    // Through the façade, which is what the settings panel calls, including the
    // status and choice projections.
    let manager = ModelManager::new(m.clone(), roots);
    let _ = manager.choices();
    let with_ui = best_of(200, || {
        let t = Instant::now();
        std::hint::black_box(manager.choices());
        t.elapsed()
    });
    println!("BENCH choices(): status + the Vec<Choice> projection in {with_ui:?}");
    assert!(
        with_ui < Duration::from_millis(50),
        "choices() regressed: {with_ui:?}"
    );
}

#[test]
#[ignore = "prints measurements; run with --nocapture --ignored"]
fn discovery_time_on_a_directory_that_looks_like_two_gigabytes() {
    // The requirement, stated as a tree: a user with a 2 GB model directory must
    // not wait seconds for a list. The files are **sparse** -- `set_len` gives a
    // file that reports 2 GiB and occupies none -- which is the honest way to
    // build the fixture, and it also makes the point: discovery's cost is a
    // function of the file *count*, not their size, because it never reads one.
    let t = TempDir::new("bench-big");
    // The directory name has to be the one the manifest claims, or there is
    // nothing to discover.
    let model_dir = t.dir("user/sravaani");
    const BIG: u64 = 2 * 1024 * 1024 * 1024;

    for name in [
        "encoder.fp16.onnx",
        "encoder.onnx",
        "decoder_joint.onnx",
        "tokenizer.model",
        "mel_window.f32",
        "mel_fb.f32",
        "preproc.json",
        "config.json",
        "model_meta.json",
        "artefacts.json",
    ] {
        let f = std::fs::File::create(model_dir.join(name)).unwrap();
        f.set_len(BIG / 4).unwrap();
    }
    for i in 0..500 {
        let f = std::fs::File::create(model_dir.join(format!("cache{i}.bin"))).unwrap();
        f.set_len(1024 * 1024).unwrap();
    }

    let m = Manifest::embedded().expect("valid");
    let roots = ModelRoots::at(t.join("user"), t.join("bundled"));
    let _ = discover::discover(&m, &roots);

    let d = best_of(50, || {
        let s = Instant::now();
        std::hint::black_box(discover::discover(&m, &roots));
        s.elapsed()
    });
    let found = discover::discover(&m, &roots);
    let on_disk = found
        .best("sravaani-1.0-onnx")
        .map_or(0, |x| x.on_disk_bytes);
    println!(
        "BENCH discovery: 504 files reporting {on_disk} bytes ({:.1} GiB) in {d:?}",
        on_disk as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    assert!(d < Duration::from_millis(100), "discovery regressed: {d:?}");

    // And the cost really is the file count, not the size: the same directory
    // with the files emptied.
    for e in std::fs::read_dir(&model_dir).unwrap().flatten() {
        if let Ok(f) = e.metadata()
            && f.len() > 1024 * 1024
        {
            let h = std::fs::File::options().write(true).open(e.path()).unwrap();
            h.set_len(0).unwrap();
        }
    }
    let empty = best_of(50, || {
        let s = Instant::now();
        std::hint::black_box(discover::discover(&m, &roots));
        s.elapsed()
    });
    println!("BENCH discovery: the same 504 files, now empty, in {empty:?}");
    assert!(
        d.as_secs_f64() < empty.as_secs_f64() * 4.0 + 0.05,
        "discovery time should not depend on file size (2 GiB: {d:?}, empty: {empty:?})"
    );
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// The child: a clean process so `PeakWorkingSetSize` means what it says.
#[test]
fn bench_child() {
    if std::env::var_os("VAANI_A7_BENCH_CHILD").is_none() {
        return;
    }
    let dir = std::env::var_os("VAANI_A7_BENCH_DIR").expect("the child needs a directory");
    let out = std::env::var_os("VAANI_A7_BENCH_OUT").expect("the child needs an output path");
    let dir = std::path::PathBuf::from(dir);

    let m = Manifest::embedded().expect("valid");
    let model = m.model("sravaani-1.0-onnx").expect("present");
    let started = Instant::now();
    let report = Verifier::new().verify(model, &dir, &CancelToken::new(), &|_| {});
    let elapsed = started.elapsed();
    let peak = vaani_models::sys::peak_working_set_bytes().unwrap_or(0);
    let body = format!(
        "bytes={} elapsed_us={} peak_rss={peak} ok={}",
        report.bytes_hashed,
        elapsed.as_micros(),
        u8::from(report.diagnosis.is_good()),
    );
    println!("A7-BENCH-CHILD {body}");
    std::fs::write(out, &body).unwrap();
}

#[test]
#[ignore = "prints measurements; run with --nocapture --ignored"]
fn verification_throughput_and_peak_memory() {
    let dir = sravaani_dir();
    if !dir.join("encoder.fp16.onnx").is_file() {
        println!("skipping: {} is not present", dir.display());
        return;
    }
    let t = TempDir::new("bench-verify");
    let out = t.join("child.txt");
    let exe = std::env::current_exe().expect("the test binary");
    let started = Instant::now();
    let status = std::process::Command::new(exe)
        .args(["--exact", "bench_child", "--nocapture", "--test-threads=1"])
        .env("VAANI_A7_BENCH_CHILD", "1")
        .env("VAANI_A7_BENCH_DIR", &dir)
        .env("VAANI_A7_BENCH_OUT", &out)
        .status()
        .expect("the child must run");
    assert!(status.success());
    let body = std::fs::read_to_string(&out).expect("the child must report");
    println!("A7-BENCH-CHILD {body}");

    let field = |name: &str| -> u64 {
        body.split_whitespace()
            .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("the child did not report {name}: {body}"))
    };
    let bytes = field("bytes");
    let micros = field("elapsed_us");
    let peak = field("peak_rss");
    let total = Duration::from_micros(micros);
    println!(
        "BENCH verify: {bytes} bytes ({:.2} GiB) in {total:?} = {:.0} MB/s \
         (whole test including the child spawn: {:?})",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        mib_per_s(bytes, total),
        started.elapsed()
    );
    println!(
        "BENCH peak RSS while verifying: {peak} bytes ({:.1} MiB)",
        peak as f64 / (1024.0 * 1024.0)
    );
    assert_eq!(field("ok"), 1, "the real artefacts must verify");
    // The file alone is 845 MiB, so a peak below that is proof the whole-file
    // read path is not being taken.
    assert!(
        peak < bytes / 2,
        "peak RSS {peak} is suspiciously close to the file size {bytes}"
    );
    assert!(
        peak < 256 * 1024 * 1024,
        "peak RSS {peak} exceeds the 256 MiB ceiling"
    );
    assert!(total > Duration::ZERO);
}

#[test]
#[ignore = "prints measurements; run with --nocapture --ignored"]
fn hashing_throughput_at_several_buffer_sizes() {
    // The buffer size is a memory/throughput trade, so it is worth knowing what
    // the default buys. 1 MiB is the shipped default; the others bound it.
    let p = sravaani_dir().join("decoder_joint.onnx");
    if !p.is_file() {
        println!("skipping: {} is not present", p.display());
        return;
    }
    let size = std::fs::metadata(&p).unwrap().len();
    for buf in [64 * 1024usize, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024] {
        let mut buffer = vec![0u8; buf];
        let runs = 5;
        let mut best = Duration::MAX;
        for _ in 0..runs {
            let t = Instant::now();
            let r = vaani_models::verify::hash_file_streaming(&p, &mut buffer, |_| true).unwrap();
            std::hint::black_box(r);
            best = best.min(t.elapsed());
        }
        println!(
            "BENCH hash: {buf} byte buffer -> {} bytes in {best:?} = {:.0} MB/s",
            size,
            mib_per_s(size, best)
        );
        assert!(best > Duration::ZERO);
    }
}
