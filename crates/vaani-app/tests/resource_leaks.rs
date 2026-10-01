//! Leaks a long session accumulates, and the sweep that recovers from them.
//!
//! Both are invisible to a single-job test. A `Vec` that only grows when a job
//! finishes looks identical after one job and after two hundred, and a work
//! directory removed on every path a test happens to take looks fine right up
//! until a process is killed. What is checked here is the *steady state*, which
//! is the only place either shows up.

use std::fs;

use vaani_app::orchestrator::Orchestrator;

/// A finished job's work directory must not survive the job.
///
/// The directory holds decoded audio and partial exports. It is created with
/// `create_dir_all` before the first stage and nothing else removed it, so
/// without cleanup a session that transcribed two hundred files left two hundred
/// directories behind for as long as the app was installed.
#[test]
fn a_finished_job_leaves_no_work_directory() {
    let root = tempfile::tempdir().expect("a temp root");
    let dir = root.path().join("job-abc");
    fs::create_dir_all(&dir).expect("create");
    fs::write(dir.join("decoded.f32"), vec![0u8; 4096]).expect("write an intermediate");
    assert!(dir.exists(), "precondition: the directory exists");

    // The cleanup is what `run_job` does on every exit; exercised directly so
    // the test does not need a model loaded.
    Orchestrator::discard_work_dir_for_test(&dir);

    assert!(
        !dir.exists(),
        "the work directory must be gone, contents included"
    );
}

/// A directory that cannot be removed must not turn into a failed job.
///
/// The job's result is decided before cleanup runs, so refusing to report a
/// completed transcription because a temporary directory was locked would trade
/// a real outcome for a cosmetic one. The cleanup is therefore best-effort, and
/// this asserts the property that makes that safe: it returns rather than
/// panicking, and leaves the path for the startup sweep.
#[test]
fn an_unremovable_work_directory_is_survivable() {
    let root = tempfile::tempdir().expect("a temp root");
    // A *file* where a directory is expected: `remove_dir_all` fails on this,
    // which is the same class of failure as a locked directory without needing
    // platform-specific locking.
    let path = root.path().join("not-a-directory");
    fs::write(&path, b"x").expect("write");

    Orchestrator::discard_work_dir_for_test(&path);

    assert!(
        path.exists(),
        "a path the cleanup could not remove is left alone rather than removed \
         by something else"
    );
}

/// The startup sweep must remove what a killed process left behind.
///
/// A job killed mid-run reaches no cleanup exit at all. Those directories are
/// decoded audio, and on a machine interrupted mid-job there can be hundreds of
/// megabytes of them with nothing else that would ever remove them.
#[test]
fn the_startup_sweep_removes_what_a_killed_process_left() {
    let root = tempfile::tempdir().expect("a temp root");

    for id in ["job-a", "job-b", "job-c"] {
        let d = root.path().join(id);
        fs::create_dir_all(&d).expect("create");
        fs::write(d.join("audio.f32"), vec![0u8; 1024]).expect("write");
    }
    // A dot-prefixed directory is how the export stage marks a transient file,
    // and the downloader keeps `.partial` in its own tree. Not a job directory.
    let dotfile_dir = root.path().join(".something-else");
    fs::create_dir_all(&dotfile_dir).expect("create");
    fs::write(dotfile_dir.join("keep"), b"x").expect("write");

    let removed = Orchestrator::sweep_stale_work_dirs(root.path());

    assert_eq!(
        removed, 3,
        "every job directory from the interrupted run must be swept"
    );
    for id in ["job-a", "job-b", "job-c"] {
        assert!(
            !root.path().join(id).exists(),
            "{id} survived the sweep and will never be cleaned up"
        );
    }
    assert!(
        dotfile_dir.exists(),
        "a dot-prefixed directory is not a job directory and must be left alone"
    );
}

/// A missing work root is a first run, not a failure.
#[test]
fn sweeping_a_directory_that_does_not_exist_is_a_no_op() {
    let root = tempfile::tempdir().expect("a temp root");
    let absent = root.path().join("never-created");

    assert_eq!(
        Orchestrator::sweep_stale_work_dirs(&absent),
        0,
        "a first run has no work root, and that is not an error"
    );
    assert!(
        !absent.exists(),
        "the sweep must not create the root either"
    );
}
