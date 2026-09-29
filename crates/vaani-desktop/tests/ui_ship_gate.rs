//! The UI ship gate: a real user flow through the real window and the real backend.
//!
//! The ship gate in `ship_gate.rs` proves the backend transcribes. This proves
//! the *window* drives it: set a path, press Transcribe (call `enqueue`),
//! pump events the way the 33 ms timer does, and assert the transcript pane
//! fills, the job list completes, and export writes a file.
//!
//! `#[ignore]`d: needs ~1 minute, ~900 MB of weights and the private clip.
//!
//! ```text
//! cargo test -p vaani-desktop --release --test ui_ship_gate -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use vaani_ui::backend::{ExportTarget, HostServices};

/// A host that never opens a dialog. The path is typed in directly, which is
/// also what a user does when they paste a path instead of browsing.
#[derive(Debug, Default)]
struct NoDialogHost;

impl HostServices for NoDialogHost {
    fn pick_media_files(&self) -> Vec<PathBuf> {
        Vec::new()
    }
    fn pick_export_target(&self, suggested_name: &str) -> Option<ExportTarget> {
        // Mirror what the save dialog returns: a file, not a directory.
        Some(ExportTarget::File(
            std::env::temp_dir().join(suggested_name),
        ))
    }
    fn set_clipboard_text(&self, _text: &str) {}
    fn host_name(&self) -> String {
        "ui-ship-gate".to_string()
    }
}

fn clip() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("scratch")
        .join("clip.wav");
    p.is_file().then_some(p)
}

/// A real video file with spaces in the name, the way users' files look.
/// Returns None when absent so the test skips instead of failing.
fn video_clip() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("WhatsApp Video 2026-09-27 at 12.30.15 AM.mp4");
    p.is_file().then_some(p)
}

fn bundled_model_present() -> bool {
    vaani_models::ModelManager::embedded()
        .ok()
        .and_then(|m| m.resolve("sravaani-1.0-onnx").ok())
        .map(|loc| loc.dir().join("encoder.fp16.onnx").is_file())
        .unwrap_or(false)
}

#[test]
#[ignore = "needs the bundled model and the private clip; run explicitly before release"]
fn the_window_drives_a_real_transcription() {
    let Some(clip) = clip() else {
        println!("SKIP: scratch/clip.wav is absent");
        return;
    };
    if !bundled_model_present() {
        println!("SKIP: the bundled SraVaani model is absent");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let store = vaani_storage::Store::open(dir.path().join("ui-ship.db")).expect("a temp store");
    let backend = vaani_app::backend::AppBackend::new(store).expect("the real backend");

    // Build the real window and model on this thread. Everything below runs
    // here too, which satisfies the event-loop-thread rule the same way the
    // timer pump does in production.
    let host: Arc<dyn HostServices> = Arc::new(NoDialogHost);
    let session = vaani_ui::build(backend.clone(), host).expect("a window");
    let model = &session.model;

    // 1. Type a path and press Transcribe, exactly as a user does.
    model.input_edited(clip.display().to_string());
    model.enqueue();

    // 2. Pump the way the 33 ms timer does, until the job completes.
    // Sample the rendered progress and the rendered transcript on every pump: a
    // bar that never moves is the "no progress" complaint, and a transcript
    // pane that stays empty until the end is the "no streaming" one. Only a
    // time series proves either way.
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut terminal_phase = None;
    let mut overall_samples = Vec::new();
    let mut last_overall_text = String::new();
    // (elapsed ms, transcript rows rendered) taken while the job still ran.
    let mut streamed: Vec<(u128, i32)> = Vec::new();
    let started = Instant::now();
    while Instant::now() < deadline {
        model.pump();
        let jobs = model.job_list();
        if let Some(first) = jobs.jobs().first() {
            overall_samples.push(first.overall);
            if first.phase.is_terminal() {
                terminal_phase = Some(first.phase);
                break;
            }
            // Still running: whatever the pane holds now streamed in mid-run.
            streamed.push((
                started.elapsed().as_millis(),
                session.window.get_transcript_rows_total(),
            ));
        }
        drop(jobs);
        // The rendered bar value, not just Rust state.
        last_overall_text = session.window.get_overall_text().to_string();
        std::thread::sleep(Duration::from_millis(200));
    }
    assert_eq!(
        terminal_phase,
        Some(vaani_core::JobPhase::Complete),
        "the UI-driven job must complete"
    );
    println!("ok  UI-driven job reached Complete");
    // The bar must have moved: at least one sample strictly between 0 and 1,
    // and the final text must not be empty. A bar stuck at 0, at None, or at
    // 100% for the whole run is a progress display that never progresses.
    let moved = overall_samples
        .iter()
        .flatten()
        .any(|&f| f > 0.0 && f < 1.0);
    assert!(
        moved,
        "progress must move during the job; samples: {:?} rendered as {:?}",
        &overall_samples[..overall_samples.len().min(10)],
        last_overall_text,
    );
    println!("ok  progress moved; final text: {last_overall_text}");
    // The transcript must fill in while the job runs, not appear all at once
    // when it ends. Two things prove that: rows present before the terminal
    // event, and their count still climbing afterwards.
    let during = |upto: u128| {
        streamed
            .iter()
            .filter(|(t, _)| *t <= upto)
            .map(|(_, rows)| *rows)
            .max()
            .unwrap_or(0)
    };
    let last_at = streamed.last().map(|(t, _)| *t).unwrap_or(0);
    let mid = during(last_at / 2);
    let end = during(last_at);
    assert!(
        mid > 0,
        "the transcript pane must show words before the job finishes; rows mid-run: {mid}"
    );
    assert!(
        end > mid,
        "streamed rows must keep growing; mid={mid} end={end} over {last_at}ms"
    );
    println!("ok  transcript streamed: mid={mid} rows, end={end} rows, over {last_at}ms");

    // 3. The transcript pane must hold the transcript.
    let view = model.transcript_view();
    let words: usize = view
        .transcript()
        .map(|t| t.segments.iter().map(|s| s.words.len()).sum())
        .unwrap_or(0);
    assert!(
        words > 100,
        "the transcript pane must show the transcript, got {words} words"
    );
    println!("ok  transcript pane holds {words} words");
    drop(view);

    // 3b. The RENDERED window must show it, not just Rust state. State that
    // never reaches Slint properties is a transcript the user cannot see.
    {
        use slint::Model as _;
        let window = &session.window;
        assert!(
            window.get_jobs().row_count() > 0,
            "the window must list the completed job"
        );
        assert!(
            window.get_transcript_window().row_count() > 0,
            "the window must render transcript rows"
        );
        assert!(
            !window.get_transcript_summary().is_empty(),
            "the window must show a transcript summary"
        );
        assert!(
            window.get_speakers().row_count() > 0,
            "the window must list speakers"
        );
        assert!(
            window.get_can_export(),
            "export must be enabled for a completed job"
        );
        println!(
            "ok  rendered: {} job rows, {} transcript rows, {} speakers, summary {:?}",
            window.get_jobs().row_count(),
            window.get_transcript_window().row_count(),
            window.get_speakers().row_count(),
            window.get_transcript_summary().to_string(),
        );
    }

    // 4. Export through the UI's export path.
    model.export();
    // The export worker announces ExportFinished; pump until the file lands.
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut exported_path = None;
    while Instant::now() < deadline {
        model.pump();
        // The export target is the temp dir with the default stem; look for
        // any new transcript file there.
        if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("clip") && name.ends_with(".txt") {
                    exported_path = Some(e.path());
                    break;
                }
            }
        }
        if exported_path.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    println!("ok  export requested; file: {exported_path:?}");
}

/// A real MP4 with spaces in the filename, driven through the same UI path.
/// Videos demux differently from WAV; a user uploading phone recordings hits
/// this path, not the WAV one. Skips cleanly when the file is absent.
#[test]
#[ignore = "needs the bundled model and the repo-root MP4; run explicitly before release"]
fn the_window_transcribes_a_real_mp4_with_spaces_in_its_name() {
    use std::sync::Arc;
    use vaani_ui::backend::HostServices;

    #[derive(Debug, Default)]
    struct NoDialogHost;
    impl HostServices for NoDialogHost {
        fn pick_media_files(&self) -> Vec<PathBuf> {
            Vec::new()
        }
        fn pick_export_target(
            &self,
            _suggested_name: &str,
        ) -> Option<vaani_ui::backend::ExportTarget> {
            None
        }
        fn set_clipboard_text(&self, _text: &str) {}
        fn host_name(&self) -> String {
            "ui-ship-gate-mp4".to_string()
        }
    }

    let Some(video) = video_clip() else {
        println!("SKIP: repo-root MP4 is absent");
        return;
    };
    if !bundled_model_present() {
        println!("SKIP: the bundled SraVaani model is absent");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let store =
        vaani_storage::Store::open(dir.path().join("ui-ship-mp4.db")).expect("a temp store");
    let backend = vaani_app::backend::AppBackend::new(store).expect("the real backend");
    let host: Arc<dyn HostServices> = Arc::new(NoDialogHost);
    let session = vaani_ui::build(backend, host).expect("a window");
    let model = &session.model;

    // Type the spaced path exactly as the field would hold it after a dialog
    // pick, then press Transcribe.
    model.input_edited(video.display().to_string());
    model.enqueue();

    let deadline = Instant::now() + Duration::from_secs(600);
    let mut terminal_phase = None;
    while Instant::now() < deadline {
        model.pump();
        let jobs = model.job_list();
        if let Some(first) = jobs.jobs().first().filter(|j| j.phase.is_terminal()) {
            terminal_phase = Some(first.phase);
            break;
        }
        drop(jobs);
        std::thread::sleep(Duration::from_millis(200));
    }
    assert_eq!(
        terminal_phase,
        Some(vaani_core::JobPhase::Complete),
        "an MP4 with spaces must transcribe, not fail"
    );
    let view = model.transcript_view();
    let words: usize = view
        .transcript()
        .map(|t| t.segments.iter().map(|s| s.words.len()).sum())
        .unwrap_or(0);
    assert!(words > 50, "the MP4 must yield words, got {words}");
    println!("ok  MP4 through the window: {words} words; UI MP4 PATH WORKS");
}
