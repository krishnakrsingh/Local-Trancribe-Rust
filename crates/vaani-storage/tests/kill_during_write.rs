//! Being killed mid-write must not cost the user anything they had already.
//!
//! # How this test actually kills a process
//!
//! It re-executes this same test binary as a child, with an environment variable
//! that puts it into "write a large transcript, then announce that you are inside
//! the transaction and wait" mode. The parent watches for the announcement file
//! and then calls [`std::process::Child::kill`], which on Windows is
//! `TerminateProcess`: the child gets no chance to unwind, no destructor runs, no
//! buffer is flushed and SQLite never gets to roll back.
//!
//! That is the point. `process::exit` would have been easier and would have
//! proved less: a clean exit path can still run destructors, and a library that
//! flushes on drop would pass a test that a real power cut or a Task Manager
//! "End task" would fail.
//!
//! # What must be true afterwards
//!
//! * every job that was **complete** before the kill is still there, with its
//!   transcript intact to the millisecond;
//! * the interrupted job is **marked**, not left claiming to be 40 % done;
//! * the transcript that was being written is either absent or complete — never
//!   half of it, which is the failure this whole design exists to prevent;
//! * the database is not corrupt and passes its own integrity check.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use vaani_core::{JobPhase, Time};
use vaani_storage::{JobRecord, Store, fixtures, testing::TempDir};

/// Set by the parent for the child process.
const ENV_MODE: &str = "VAANI_A6_CHILD";
/// Where the child announces that it is inside the write transaction.
const ENV_SENTINEL: &str = "VAANI_A6_SENTINEL";
/// The database both processes use.
const ENV_DB: &str = "VAANI_A6_DB";

/// The one test that runs in the child. It is `#[ignore]`d in normal runs so a
/// plain `cargo test` does not spawn it by accident; the parent runs it by name.
#[test]
#[ignore = "spawned by kill_during_write_is_survivable"]
fn child_writes_until_it_is_killed() {
    if std::env::var(ENV_MODE).is_err() {
        return;
    }
    let db = PathBuf::from(std::env::var(ENV_DB).unwrap());
    let sentinel = PathBuf::from(std::env::var(ENV_SENTINEL).unwrap());
    let store = Store::open(&db).unwrap();

    // A big transcript, so the write takes long enough to be interrupted
    // somewhere interesting rather than before it starts.
    let transcript = fixtures::transcript(0xBEEF, 4_000);

    // Announce *before* the write, then write: the parent may kill us at any
    // point, including before the first row is inserted, which is also a case
    // worth surviving.
    {
        let mut f = std::fs::File::create(&sentinel).unwrap();
        f.write_all(b"writing").unwrap();
        f.sync_all().unwrap();
    }
    loop {
        // Re-writing the same transcript is one transaction each time, so the
        // parent can kill us in the middle of any of them.
        let _ = store.replace_transcript("victim", &transcript);
        std::fs::write(&sentinel, b"writing").unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn kill_during_write_is_survivable() {
    if std::env::var(ENV_MODE).is_ok() {
        return; // the child does not run the parent's test
    }
    // Three rounds, because *where* the kill lands is not controllable. Each
    // round asserts the same invariants at a different, effectively random,
    // point in the write; a single round would only ever prove it once.
    for round in 0..3u64 {
        one_round(round);
    }
}

fn one_round(round: u64) {
    let dir = TempDir::new(&format!("kill{round}"));
    let db = dir.join("v.db");
    let sentinel = dir.join("sentinel");

    // --- Before the kill: three finished jobs, of real size. -----------------
    let mut before: Vec<(String, vaani_core::Transcript)> = Vec::new();
    {
        let store = Store::open(&db).unwrap();
        for (i, name) in ["monday.m4a", "tuesday.wav", "wednesday.mp4"]
            .into_iter()
            .enumerate()
        {
            let id = format!("done-{i}");
            let mut job = JobRecord::new(&id, dir.join(name));
            job.phase = JobPhase::Complete;
            job.model = Some("sravaani-1.0".into());
            store.upsert_job(&job).unwrap();
            let t = fixtures::transcript(100 + i as u64 + round * 10, 300);
            store.replace_transcript(&id, &t).unwrap();
            before.push((id, t));
        }
        // The job the child will be working on.
        let mut victim = JobRecord::new("victim", dir.join("victim.wav"));
        victim.phase = JobPhase::Asr;
        store.upsert_job(&victim).unwrap();
        store
            .replace_transcript("victim", &fixtures::transcript(1, 2))
            .unwrap();
        store.close().unwrap();
    }

    // --- Start the child and kill it mid-transaction. -----------------------
    let mut child = spawn_child(&db, &sentinel);
    let began = Instant::now();
    wait_for(&sentinel, Duration::from_secs(60));
    // Give it a moment to actually be inside a write transaction rather than
    // between two of them.
    std::thread::sleep(Duration::from_millis(50));
    child.kill().expect("the child must still be running");
    let status = child.wait().expect("the child must be reapable");
    assert!(
        !status.success(),
        "the child was supposed to be killed, not to finish: {status}"
    );
    assert!(
        began.elapsed() < Duration::from_secs(90),
        "the child never reached its write transaction"
    );

    // --- After the kill. ----------------------------------------------------
    let store = Store::open(&db).expect("the database must still open");

    // 1. Everything that was finished is still finished, byte for byte.
    for (id, t) in &before {
        let job = store
            .load_job(id)
            .unwrap()
            .unwrap_or_else(|| panic!("{id} was lost"));
        assert_eq!(job.phase, JobPhase::Complete, "{id} lost its phase");
        let back = store
            .load_transcript(id)
            .unwrap_or_else(|e| panic!("{id} lost its transcript: {e:?}"));
        assert!(
            fixtures::diff(t, &back).is_none(),
            "{id} changed across the kill"
        );
    }

    // 2. The interrupted job is marked, not stuck.
    let victim = store.load_job("victim").unwrap().expect("the victim job");
    assert!(
        victim.phase.is_terminal(),
        "an interrupted job must not stay in {:?}",
        victim.phase
    );
    assert_eq!(victim.phase, JobPhase::Failed);
    let why = victim.error.expect("an interrupted job must say why");
    assert!(
        why.message.contains("closed while this job was running"),
        "{why:?}"
    );

    // 3. Its transcript is all-or-nothing: either the two segments from before
    //    the kill, or the four thousand from the interrupted write, or nothing.
    //    Never a mixture, and never a partial segment.
    match store.try_load_transcript("victim").unwrap() {
        None => {}
        Some(t) => {
            assert!(
                t.segments.len() == 2 || t.segments.len() == 4_000,
                "a half-written transcript survived: {} segments",
                t.segments.len()
            );
            for seg in &t.segments {
                let words = store
                    .with_read(|conn| {
                        let map =
                            |e: rusqlite::Error| vaani_storage::error::from_sqlite(e, store.path());
                        conn.query_row(
                            "SELECT COUNT(*) FROM words WHERE segment_id = ?1",
                            [&seg.id],
                            |r| r.get::<_, i64>(0),
                        )
                        .map_err(map)
                    })
                    .unwrap();
                assert_eq!(
                    words,
                    seg.words.len() as i64,
                    "segment {:?} has {words} stored words but {} in the transcript",
                    seg.id,
                    seg.words.len()
                );
            }
        }
    }

    // 4. The database is intact, and says so.
    let report = store.integrity_check().unwrap();
    assert!(
        report.ok,
        "the database is damaged after the kill: {:?}",
        report.messages
    );
    let stats = store.stats().unwrap();
    assert_eq!(stats.job_count, 4, "a job went missing");
    assert!(stats.word_count > 0);
}

/// Launch this test binary again as the child.
fn spawn_child(db: &Path, sentinel: &Path) -> std::process::Child {
    let exe = std::env::current_exe().expect("the test binary's own path");
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--ignored")
        .arg("--exact")
        .arg("child_writes_until_it_is_killed")
        .arg("--nocapture")
        .env(ENV_MODE, "1")
        .env(ENV_DB, db)
        .env(ENV_SENTINEL, sentinel)
        // The child's own output is noise in the parent's test log.
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.spawn().expect("the child must start")
}

fn wait_for(path: &Path, within: Duration) {
    let began = Instant::now();
    while began.elapsed() < within {
        if path.exists() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("the child never announced that it was writing");
}

#[test]
fn a_killed_process_leaves_no_temporary_files_behind() {
    // The other half of "safe": nothing half-written should be sitting in the
    // data directory for a user to find.
    let dir = TempDir::new("kill-files");
    let db = dir.join("v.db");
    let sentinel = dir.join("sentinel");
    {
        let store = Store::open(&db).unwrap();
        store
            .upsert_job(&JobRecord::new("victim", dir.join("a.wav")))
            .unwrap();
    }
    let mut child = spawn_child(&db, &sentinel);
    wait_for(&sentinel, Duration::from_secs(60));
    std::thread::sleep(Duration::from_millis(50));
    child.kill().unwrap();
    child.wait().unwrap();

    let leftovers: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("tmp") || n.contains("partial") || n.contains("bak"))
        .collect();
    assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    // The write-ahead log is expected — it is SQLite's, and it is replayed on
    // the next open.
    assert!(db.exists());
}

#[test]
fn a_time_value_at_the_extremes_survives_a_reopen() {
    // A millisecond count is an i64. The extremes are the ones a rounding bug
    // produces, and they must not become a constraint violation or a wrapped
    // value on the way to disk.
    let dir = TempDir::new("kill-extremes");
    let db = dir.join("v.db");
    {
        let store = Store::open(&db).unwrap();
        store
            .upsert_job(&JobRecord::new("j", dir.join("a.wav")))
            .unwrap();
        let mut t = fixtures::empty();
        for (i, ms) in [i64::MIN / 2, -1, 0, 1, i64::MAX / 2]
            .into_iter()
            .enumerate()
        {
            let mut seg = vaani_core::Segment::new(Time(ms), Time(ms));
            seg.id = format!("s{i}");
            seg.words = vec![vaani_core::Word::new("x", Time(ms), Time(ms))];
            t.segments.push(seg);
        }
        store.replace_transcript("j", &t).unwrap();
    }
    let store = Store::open(&db).unwrap();
    let back = store.load_transcript("j").unwrap();
    for (i, ms) in [i64::MIN / 2, -1, 0, 1, i64::MAX / 2]
        .into_iter()
        .enumerate()
    {
        assert_eq!(back.segments[i].start.as_millis(), ms, "segment {i}");
        assert_eq!(back.segments[i].words[0].end.as_millis(), ms, "word {i}");
    }
}
