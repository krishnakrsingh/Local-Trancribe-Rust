//! Measured numbers at the scale a real meeting produces.
//!
//! Ignored by default because it takes a minute and writes a database; run it
//! with:
//!
//! ```text
//! cargo test -p vaani-export --test real_size -- --ignored --nocapture
//! ```
//!
//! or through `scripts/a6/bench.ps1`.
//!
//! # What is real here and what is not
//!
//! **Real:** the private `scratch/clip.wav` — a ~13 minute excerpt of a real
//! meeting — is read (never copied, never modified, never printed) to measure
//! the one storage operation that touches the media file itself: hashing it for
//! the duplicate-transcription check. Its size and hash latency are reported as
//! aggregate numbers.
//!
//! **Not real:** the transcript. Running the ASR pipeline is A2's and A5's job
//! and their measurements, and this crate must not depend on a half-finished
//! pipeline to be tested. So the transcript is **generated at the scale A5
//! measured** — 1,800+ segments and ~20,000 words for a 13-minute meeting — and
//! every number below is a real measurement of storing, loading, searching and
//! exporting *that*. The shape is right; the words are not real, and no word from
//! the recording is used, derived or guessed at anywhere in this file.
//!
//! Every figure printed is a count, a byte total or a duration. No transcript
//! text of any kind is printed, and the database is deleted when the test ends.
use std::path::Path;
use std::time::{Duration, Instant};

use vaani_storage::{
    ExportRecord, JobFilter, JobRecord, SearchQuery, Store, fixtures, testing::TempDir,
};

/// A5's measured shape for a 13-minute meeting: 1,800+ segments, ~20,000 words.
const MEETING_SEGMENTS: usize = 1_850;
/// A5's measured word count for the same meeting.
const MEETING_WORDS: usize = 20_000;
/// A few hundred jobs is a heavy user's history.
const CORPUS_JOBS: usize = 300;

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1000.0 * 100.0).round() / 100.0
}

fn mib(bytes: u64) -> f64 {
    (bytes as f64 / 1_048_576.0 * 100.0).round() / 100.0
}

/// A transcript with the measured shape: ~1,850 segments and ~20,000 words.
///
/// The generator produces short segments, so the word list of each segment is
/// padded with synthetic tokens until the document reaches the word count A5
/// measured. The padding is filler chosen for *this* benchmark: it is not
/// derived from, inspired by, or informed by the private recording, and it is
/// never printed. Its only job is to make the row count, the index size and the
/// file sizes the ones a real meeting produces.
fn meeting() -> vaani_core::Transcript {
    let mut t = fixtures::transcript(0xA6, MEETING_SEGMENTS);
    t.duration = Some(vaani_core::MediaLength::from_millis(783_000));
    let target_words = MEETING_WORDS;
    let per_segment = target_words / MEETING_SEGMENTS;
    for (i, seg) in t.segments.iter_mut().enumerate() {
        while seg.words.len() < per_segment {
            let n = seg.words.len();
            let start = seg.start.as_millis() + 20 * n as i64;
            seg.words.push(vaani_core::Word::new(
                format!("filler{n}"),
                vaani_core::Time(start),
                vaani_core::Time(start + 19),
            ));
        }
        let _ = i;
    }
    t
}

#[test]
#[ignore = "writes a database and takes about a minute; run via scripts/a6/bench.ps1"]
fn measure_a_real_size_transcript() {
    println!("\n=== vaani-storage / vaani-export at real-meeting scale ===\n");
    let dir = TempDir::new("bench");
    let db = dir.join("bench.db");

    // --- The one measurement that touches the real recording ----------------
    // Read-only, aggregate numbers only. The file is a private meeting, so
    // nothing about its contents is printed, derived or stored.
    // Resolved from the manifest directory so the test does not depend on the
    // working directory cargo happens to use.
    let clip = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scratch/clip.wav");
    if clip.exists() {
        match std::fs::metadata(&clip) {
            Ok(meta) => {
                let began = Instant::now();
                let hash = vaani_storage::hash::hash_file(&clip).unwrap();
                println!(
                    "real recording (scratch/clip.wav, read-only):\n  \
                     size            {:>12} bytes ({:.1} MiB)\n  \
                     sha256 (dedup)  {:>12.1} ms\n  \
                     hash            {}...",
                    meta.len(),
                    mib(meta.len()),
                    ms(began.elapsed()),
                    &hash[..16]
                );
            }
            Err(e) => println!("real recording: could not stat it ({e})"),
        }
    } else {
        println!("real recording: scratch/clip.wav is not present in this checkout");
    }

    // --- Store one real-size transcript -------------------------------------
    let transcript = meeting();
    let words: usize = transcript.segments.iter().map(|s| s.words.len()).sum();
    println!(
        "\ntranscript shape:\n  segments        {:>12}\n  words           {:>12}\n  speakers        {:>12}",
        transcript.segments.len(),
        words,
        transcript.speakers.len()
    );

    let store = Store::open(&db).unwrap();
    let mut job = JobRecord::new("bench-0", dir.join("bench.m4a"));
    job.phase = vaani_core::JobPhase::Complete;
    job.duration_ms = Some(783_000);
    store.upsert_job(&job).unwrap();

    let began = Instant::now();
    let stats = store.replace_transcript("bench-0", &transcript).unwrap();
    let insert = began.elapsed();
    println!(
        "\nwrite:\n  \
         replace_transcript  {:>9.1} ms  ({} segments, {} words, one transaction)\n  \
         per segment         {:>9.3} ms",
        ms(insert),
        stats.segments,
        stats.words,
        insert.as_secs_f64() * 1000.0 / stats.segments as f64
    );

    let began = Instant::now();
    let loaded = store.load_transcript("bench-0").unwrap();
    let load = began.elapsed();
    assert!(
        fixtures::diff(&transcript, &loaded).is_none(),
        "the transcript did not survive the round trip at scale"
    );
    println!(
        "read:\n  \
         load_transcript    {:>9.1} ms  ({} words, exact round trip verified)",
        ms(load),
        words
    );

    let began = Instant::now();
    let _summary = store.transcript_summary("bench-0").unwrap();
    let summary = began.elapsed();
    println!(
        "  transcript_summary  {:>9.1} ms  (no words loaded)",
        ms(summary)
    );

    let began = Instant::now();
    let page = store.list_jobs(&JobFilter::default(), 50, 0).unwrap();
    let list = began.elapsed();
    println!(
        "  list_jobs(50)       {:>9.1} ms  ({} jobs match)",
        ms(list),
        page.total
    );

    // --- Search ------------------------------------------------------------
    let needle = fixtures::any_word(&transcript);
    for query in [needle.clone(), "transcription".into(), "quick brown".into()] {
        let mut samples = Vec::new();
        let mut hits = 0;
        for _ in 0..20 {
            let began = Instant::now();
            let r = store
                .search(&SearchQuery::new(&query).with_limit(50))
                .unwrap();
            samples.push(began.elapsed());
            hits = r.hits.len();
        }
        samples.sort();
        println!(
            "search  {:>9.1} ms median  {:>8.3} ms min  ({:>3} hits, limit 50, {} jobs)",
            ms(samples[samples.len() / 2]),
            ms(samples[0]),
            hits,
            store.stats().unwrap().job_count
        );
    }

    // --- On disk, for this one transcript ------------------------------------
    store.checkpoint().unwrap();
    let one = store.stats().unwrap();
    println!(
        "\non disk, one real-size transcript:\n  \
         database            {:>12} bytes ({:.1} MiB)\n  \
         bytes per word       {:>12.2} bytes\n  \
         bytes per segment    {:>12.1} bytes",
        one.database_bytes,
        mib(one.database_bytes),
        one.database_bytes as f64 / one.word_count as f64,
        one.database_bytes as f64 / one.segment_count as f64
    );

    // --- A corpus of a few hundred jobs -------------------------------------
    println!("\ncorpus of {CORPUS_JOBS} jobs:");
    let began = Instant::now();
    for i in 1..CORPUS_JOBS {
        let id = format!("bench-{i}");
        let mut j = JobRecord::new(&id, dir.join(format!("m{i}.m4a")))
            .with_content_hash(format!("{i:064x}"));
        j.phase = vaani_core::JobPhase::Complete;
        j.created_at_ms = 1_700_000_000_000 + i as i64 * 1_000;
        store.upsert_job(&j).unwrap();
        // A tenth of the size of the real one, so the corpus is a realistic mix
        // rather than 300 copies of the same meeting.
        let t = fixtures::transcript(0xB000 + i as u64, MEETING_SEGMENTS / 10);
        store.replace_transcript(&id, &t).unwrap();
    }
    let build = began.elapsed();
    let stats = store.stats().unwrap();
    println!(
        "  build              {:>9.1} s    for {} jobs, {} segments, {} words",
        build.as_secs_f64(),
        CORPUS_JOBS,
        stats.segment_count,
        stats.word_count
    );

    let mut samples = Vec::new();
    let mut hits = 0;
    for _ in 0..20 {
        let began = Instant::now();
        let r = store
            .search(&SearchQuery::new("transcription").with_limit(50))
            .unwrap();
        samples.push(began.elapsed());
        hits = r.hits.len();
    }
    samples.sort();
    println!(
        "  search median     {:>9.1} ms  ({hits} hits over {CORPUS_JOBS} jobs)",
        ms(samples[samples.len() / 2])
    );
    let began = Instant::now();
    let page = store.list_jobs(&JobFilter::default(), 50, 0).unwrap();
    println!(
        "  list_jobs(50)      {:>9.1} ms  ({} total)",
        ms(began.elapsed()),
        page.total
    );
    let began = Instant::now();
    let filtered = store
        .list_jobs(
            &JobFilter::default()
                .with_phases([vaani_core::JobPhase::Complete])
                .with_name("m12"),
            50,
            0,
        )
        .unwrap();
    println!(
        "  filtered list      {:>9.1} ms  ({} match)",
        ms(began.elapsed()),
        filtered.total
    );

    // --- On disk, for the whole corpus ---------------------------------------
    store.checkpoint().unwrap();
    let stats = store.stats().unwrap();
    println!(
        "\non disk, the whole corpus:\n  \
         database            {:>12} bytes ({:.1} MiB)\n  \
         wal (after checkpoint){:>9} bytes\n  \
         bytes per word       {:>12.2} bytes\n  \
         bytes per segment    {:>12.1} bytes",
        stats.database_bytes,
        mib(stats.database_bytes),
        stats.wal_bytes,
        stats.database_bytes as f64 / stats.word_count as f64,
        stats.database_bytes as f64 / stats.segment_count as f64
    );

    // --- Export ------------------------------------------------------------
    println!("\nexport of the real-size transcript:");
    let out = TempDir::new("bench-out");
    let options = vaani_export::ExportOptions::default().with_source_name("bench.m4a");
    for f in vaani_export::ExportFormat::ALL {
        let target = vaani_export::ExportTarget::Directory(out.path().to_path_buf());
        let began = Instant::now();
        let written = vaani_export::export_to_file(&transcript, f, &target, &options).unwrap();
        let took = began.elapsed();
        let record = ExportRecord::new(Some("bench-0"), &written.path, f.as_str(), written.bytes);
        store.record_export(&record).unwrap();
        let cues = if written.cues > 0 {
            format!(", {} cues", written.cues)
        } else {
            String::new()
        };
        println!(
            "  {:<9} {:>10.1} ms  {:>10} bytes ({:.2} MiB){}",
            f.as_str(),
            ms(took),
            written.bytes,
            mib(written.bytes),
            cues
        );
    }
    let total: u64 = vaani_export::ExportFormat::ALL
        .iter()
        .filter_map(|f| std::fs::metadata(out.path().join(format!("bench.{}", f.extension()))).ok())
        .map(|m| m.len())
        .sum();
    println!(
        "  {:<9} {:>10}        {:>10} bytes ({:.2} MiB) for all five",
        "total",
        "",
        total,
        mib(total)
    );

    let report = store.integrity_check().unwrap();
    println!(
        "\nintegrity check: {}",
        if report.ok { "ok" } else { "FAILED" }
    );
    assert!(report.ok);
    println!("\n=== end ===\n");
}
