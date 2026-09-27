//! Two installers must not install the same model at once.
//!
//! Two levels, because the requirement is two levels:
//!
//! * **threads**, in one process, which is what two panels of one window do;
//! * **processes**, by re-invoking this very test binary with `--exact`, which is
//!   what the UI and the headless CLI are. A thread test cannot see a bug where
//!   the lock is only honoured in memory, and a process test needs a real
//!   `CreateFile` on the lock file from a second process to be meaningful.
//!
//! The child is selected by an environment variable and does its work through
//! exactly the public API a second window would use: `ModelManager::install`.

mod support;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use support::{Body, TempDir, TestServer, ramp};
use vaani_models::download::{DownloadPolicy, Downloader, FixedFreeSpace};
use vaani_models::lock::InstallLock;
use vaani_models::manifest::Manifest;
use vaani_models::progress::ProgressLog;
use vaani_models::{CancelToken, InstallOptions, ModelManager, ModelRoots};

const CHILD_ENV: &str = "VAANI_A7_RACE_CHILD";
const CHILD_ROOT_ENV: &str = "VAANI_A7_RACE_ROOT";
const CHILD_URL_ENV: &str = "VAANI_A7_RACE_URL";
const CHILD_OUT_ENV: &str = "VAANI_A7_RACE_OUT";

/// The child's verdict, written where the parent can read it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ChildOutcome {
    /// "won" or "lost" or "other".
    verdict: String,
    /// The user-facing error code, when it lost.
    code: Option<String>,
    /// The message, for a human reading a failure.
    detail: String,
}

fn policy() -> DownloadPolicy {
    DownloadPolicy {
        // Four attempts, zero delay: the product retries a transport error, and
        // the loopback server these tests use has socket races a real CDN does
        // not. One attempt would make this a test of the harness.
        max_attempts: 4,
        base_delay: Duration::ZERO,
        max_delay: Duration::ZERO,
        reserve_bytes: 0,
        chunk_bytes: 4096,
        ..DownloadPolicy::default()
    }
}

fn manifest_for(url: &str, len: u64, sha: &str) -> Manifest {
    Manifest::parse(&format!(
        r#"{{"schema_version":1,"models":[{{
          "id":"m1","engine_id":"e1","version":"1","display_name":"M1",
          "description":"d","task":"asr","architecture":"a","access":"open",
          "license":{{"spdx":"MIT","name":"MIT","url":"https://example.org/l"}},
          "source":{{"kind":"direct","url_template":"{url}{{path}}"}},
          "languages":["hi"],"sample_rates":[16000],"size_bytes":{len},
          "files":[{{"path":"weights.bin","role":"weights","bytes":{len},"sha256":"{sha}"}}]
        }}]}}"#
    ))
    .expect("the generated manifest must be valid")
}

fn manager(user: &Path, bundled: &Path, url: &str, len: u64, sha: &str) -> ModelManager {
    ModelManager::new(manifest_for(url, len, sha), ModelRoots::at(user, bundled)).with_downloader(
        Downloader::with_policy(policy(), Arc::new(FixedFreeSpace(Some(1 << 30)))),
    )
}

fn sha_of(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

// ---------------------------------------------------------------------------
// The child process
// ---------------------------------------------------------------------------

/// Runs only when re-invoked by [`two_processes_racing_for_one_model`]. The body
/// is the whole public path a second window would take.
#[test]
fn race_child() {
    let Some(root) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    let user = Path::new(&root).join("user");
    let url = std::env::var(CHILD_URL_ENV).expect("the child needs a url");
    let out = std::env::var(CHILD_OUT_ENV).expect("the child needs an output path");
    let data = ramp(256 * 1024);
    let len = data.len() as u64;
    let sha = sha_of(&data);
    let m = manager(&user, &Path::new(&root).join("bundled"), &url, len, &sha);

    // Start together: the parent waits for every child to report readiness
    // before letting any of them proceed, so the race is real rather than
    // staged. Without this the first process would usually finish its install
    // before the second had even connected.
    let ready_dir = Path::new(&root).join("ready");
    std::fs::write(ready_dir.join(std::process::id().to_string()), b"1").unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while std::fs::read_dir(&ready_dir)
        .map(|d| d.count())
        .unwrap_or(0)
        < 2
    {
        assert!(
            Instant::now() < deadline,
            "the other child never reported ready"
        );
        std::thread::sleep(Duration::from_millis(2));
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let log = ProgressLog::new();
    let sink = log.sink();
    let outcome = match runtime.block_on(m.install(
        "m1",
        &InstallOptions::default(),
        &CancelToken::new(),
        &sink,
    )) {
        Ok(report) => ChildOutcome {
            verdict: if report.already_installed {
                "lost".into()
            } else {
                "won".into()
            },
            code: None,
            detail: format!("installed {} file(s)", report.installed.len()),
        },
        Err(e) => ChildOutcome {
            verdict: "lost".into(),
            code: Some(e.code().as_str().to_string()),
            // The full chain, so a failure here says *why* rather than just what.
            detail: e.debug_chain(),
        },
    };
    std::fs::write(&out, serde_json::to_vec(&outcome).unwrap()).unwrap();
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn eight_threads_racing_for_one_model_produce_exactly_one_winner() {
    // The model is served slowly enough that an install takes long enough for
    // the others to collide with it, which is the only way this test means
    // anything: eight threads that each finish before the next starts would
    // never contend.
    // 256 KiB, matching the size the child expects: the client refuses a body
    // larger than the manifest says, which is its own guard against a server
    // serving the wrong object.
    let data = ramp(256 * 1024);
    let s = TestServer::start(std::collections::BTreeMap::from([(
        "weights.bin".to_string(),
        Body::slow(data.clone(), 1),
    )]));
    let t = TempDir::new("race-threads");
    let user = t.dir("user");
    let sha = sha_of(&data);
    let m = std::sync::Arc::new(manager(
        &user,
        &t.join("bundled"),
        &s.url(""),
        data.len() as u64,
        &sha,
    ));

    // `tokio::sync::Barrier`, not `std::sync::Barrier`: eight tasks on four
    // worker threads, and a blocking barrier would occupy every worker waiting
    // for tasks that can never be polled.
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(8));
    let mut set = tokio::task::JoinSet::new();
    for _ in 0..8 {
        let m = std::sync::Arc::clone(&m);
        let barrier = std::sync::Arc::clone(&barrier);
        set.spawn(async move {
            barrier.wait().await;
            m.install(
                "m1",
                &InstallOptions::default(),
                &CancelToken::new(),
                &|_| {},
            )
            .await
        });
    }

    let mut winners = 0;
    let mut losers = 0;
    let mut loser_messages = Vec::new();
    while let Some(r) = set.join_next().await {
        match r.expect("no task may panic") {
            Ok(report) => {
                if report.already_installed {
                    losers += 1;
                } else {
                    winners += 1;
                }
            }
            Err(e) => {
                losers += 1;
                loser_messages.push(format!("{}: {}", e.code(), e.user.title));
            }
        }
    }
    assert_eq!(winners, 1, "exactly one install may do the work");
    assert_eq!(losers, 7);
    // The losers must be told clearly, and every one of them must be the same
    // kind of message: "someone else is already doing this".
    for m in &loser_messages {
        assert!(
            m.contains("Already being installed") || m.contains("already"),
            "a loser's message must name the contention: {m}"
        );
    }
    assert!(
        !loser_messages.is_empty(),
        "at least one loser must have got a clear refusal, not a silent success"
    );

    // And the directory ends up valid, with the real bytes in it.
    let installed = user.join("m1/weights.bin");
    assert_eq!(
        std::fs::read(&installed).unwrap(),
        data,
        "the winner's bytes must be the real ones"
    );
    let m = manager(
        &user,
        &t.join("bundled"),
        &s.url(""),
        data.len() as u64,
        &sha,
    );
    assert!(m.discover().is_installed("m1"));
    let v = m
        .verify("m1", &CancelToken::new(), &|_| {})
        .expect("verify must run");
    assert!(v.is_ok(), "{:?}", v.files);
    // Nothing is left behind.
    // The `.partial` directory survives, empty: it is created up front so the
    // download has somewhere to write, and removing it would be a second
    // filesystem operation with its own failure mode. What must not survive is
    // a *file* in it.
    let leftovers: Vec<String> = std::fs::read_dir(user.join("m1/.partial"))
        .map(|d| {
            d.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        leftovers.is_empty(),
        "a partial download was left behind: {leftovers:?}"
    );
    assert!(
        !user.join(".vaani-locks/m1.lock").exists(),
        "the lock is released"
    );
}

#[test]
fn two_locks_in_one_process_still_produce_one_winner() {
    // The pure lock, with no download machinery, at the thread level. Kept
    // separate because it is the mechanism, and the test above is the behaviour.
    let t = TempDir::new("race-lock");
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(16));
    let held = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_concurrent = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..16 {
        let t = t.path().to_path_buf();
        let barrier = std::sync::Arc::clone(&barrier);
        let held = std::sync::Arc::clone(&held);
        let max_concurrent = std::sync::Arc::clone(&max_concurrent);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            if let Ok(lock) = InstallLock::acquire(&t, "m1", "m1") {
                let now = held.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                max_concurrent.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(20));
                held.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                drop(lock);
                true
            } else {
                false
            }
        }));
    }
    let acquired: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(acquired.iter().filter(|x| **x).count(), 1);
    assert_eq!(
        max_concurrent.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "two threads must never hold the lock at once"
    );
}

// ---------------------------------------------------------------------------
// Processes
// ---------------------------------------------------------------------------

#[test]
fn two_processes_racing_for_one_model_produce_exactly_one_winner() {
    // Real contention across a real process boundary. Both children are spawned
    // before either is released, so the `create_new` on the lock file is what
    // decides the winner, not the order they happen to reach it in.
    // 256 KiB, matching the size the child expects: the client refuses a body
    // larger than the manifest says, which is its own guard against a server
    // serving the wrong object.
    let data = ramp(256 * 1024);
    let s = TestServer::start(std::collections::BTreeMap::from([(
        "weights.bin".to_string(),
        Body::slow(data.clone(), 2),
    )]));
    let t = TempDir::new("race-proc");
    let root = t.path().to_path_buf();
    let user = t.dir("user");
    let ready = t.dir("ready");
    let exe = std::env::current_exe().expect("the test binary's own path");

    let mut children = Vec::new();
    for i in 0..2 {
        let out = t.join(&format!("out{i}.json"));
        let c = std::process::Command::new(&exe)
            .args(["--exact", "race_child", "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, &root)
            .env(CHILD_ROOT_ENV, &user)
            .env(CHILD_URL_ENV, s.url(""))
            .env(CHILD_OUT_ENV, &out)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("the child must start");
        children.push((c, out));
    }

    // Both children write a ready file and spin until there are two, so neither
    // can finish its install before the other has started.
    let deadline = Instant::now() + Duration::from_secs(30);
    while std::fs::read_dir(&ready).map(|d| d.count()).unwrap_or(0) < 2 {
        assert!(Instant::now() < deadline, "the children never became ready");
        std::thread::sleep(Duration::from_millis(5));
    }

    let mut outcomes = Vec::new();
    for (mut c, out) in children {
        let status = c.wait().expect("the child must exit");
        let mut buf = Vec::new();
        if let Some(mut pipe) = c.stdout.take() {
            use std::io::Read;
            let _ = pipe.read_to_end(&mut buf);
        }
        let stdout = String::from_utf8_lossy(&buf).into_owned();
        assert!(status.success(), "a child failed: {status}\n{stdout}");
        let body = std::fs::read(&out)
            .unwrap_or_else(|e| panic!("a child wrote no outcome: {e}\n{stdout}"));
        outcomes.push(serde_json::from_slice::<ChildOutcome>(&body).expect("valid outcome JSON"));
    }

    let winners = outcomes.iter().filter(|o| o.verdict == "won").count();
    let losers = outcomes.iter().filter(|o| o.verdict == "lost").count();
    assert_eq!(
        winners + losers,
        2,
        "both children must reach a verdict: {outcomes:?}"
    );
    assert_eq!(winners, 1, "exactly one process may install: {outcomes:?}");

    // The loser must have been told why, in the shared taxonomy.
    let loser = outcomes.iter().find(|o| o.verdict == "lost").unwrap();
    assert_eq!(
        loser.code.as_deref(),
        Some("storage_failed"),
        "the loser must get a coded refusal, not a crash: {loser:?}"
    );
    assert!(
        loser.detail.contains("already being installed"),
        "the loser must be told another window has it: {}",
        loser.detail
    );

    // And the model directory ends up valid, with the real bytes.
    assert_eq!(std::fs::read(user.join("m1/weights.bin")).unwrap(), data);
    let m = manager(
        &user,
        &t.join("bundled"),
        &s.url(""),
        data.len() as u64,
        &sha_of(&data),
    );
    assert!(m.discover().is_installed("m1"));
    let v = m
        .verify("m1", &CancelToken::new(), &|_| {})
        .expect("verify");
    assert!(v.is_ok(), "{:?}", v.files);
    let leftovers: Vec<String> = std::fs::read_dir(user.join("m1/.partial"))
        .map(|d| {
            d.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        leftovers.is_empty(),
        "a partial download was left behind: {leftovers:?}"
    );
    assert!(!user.join(".vaani-locks/m1.lock").exists());
}

#[test]
fn a_lock_left_by_a_dead_process_does_not_block_the_next_install() {
    // The crash case, and the reason staleness is decided by trying to open the
    // lock rather than by reading a timestamp: a handle dies with its process,
    // so "nobody can open it" is proof, and a pid heuristic is not.
    let t = TempDir::new("race-stale");
    std::fs::create_dir_all(t.join(".vaani-locks")).unwrap();
    std::fs::write(t.join(".vaani-locks/m1.lock"), "{\"pid\":999999}").unwrap();
    let lock = InstallLock::acquire(t.path(), "m1", "m1").expect("a stale lock must not deadlock");
    assert_eq!(lock.holder().pid, std::process::id());
}
