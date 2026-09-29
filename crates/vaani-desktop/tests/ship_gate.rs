//! The ship gate: the exact path a user takes through the window, without the
//! window.
//!
//! The UI calls `enqueue`, watches `subscribe` for a terminal phase, then
//! reads `transcript`, `speakers` and `export`. This test does the same calls
//! in the same order against the real `AppBackend`, the real bundled model and
//! a real recording. If it passes, the only untested seam is pixels.
//!
//! `#[ignore]`d because it needs ~1 minute, ~900 MB of weights and a real
//! recording. Run it explicitly before every release:
//!
//! ```text
//! cargo test -p vaani-desktop --release --test ship_gate -- --ignored --nocapture
//! ```
//!
//! It skips cleanly (exit 0, one line saying why) when the model or the clip
//! is absent, so it never fails a machine that cannot run it. A skip is not a
//! pass, and the output says which it was.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use vaani_ui::backend::{
    ControllerEvent, ExportFormat, ExportRequest, ExportTarget, JobController, JobId,
};

/// The real clip, resolved from the repository root. Never copied, never
/// committed; read in place.
fn clip() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("scratch")
        .join("clip.wav");
    p.is_file().then_some(p)
}

/// The bundled model directory, resolved the way the app resolves it.
fn bundled_model_present() -> bool {
    vaani_models::ModelManager::embedded()
        .ok()
        .and_then(|m| m.resolve("sravaani-1.0-onnx").ok())
        .map(|loc| loc.dir().join("encoder.fp16.onnx").is_file())
        .unwrap_or(false)
}

fn wait_for_terminal(
    rx: &crossbeam_channel::Receiver<ControllerEvent>,
    id: &JobId,
    timeout: Duration,
) -> Option<vaani_core::JobPhase> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return None;
        }
        match rx.recv_timeout(remaining) {
            Ok(ev) => {
                let (eid, phase) = match &ev {
                    ControllerEvent::JobAdded(s) | ControllerEvent::JobChanged(s) => {
                        (s.id.clone(), Some(s.phase))
                    }
                    _ => continue,
                };
                if &eid == id
                    && let Some(p) = phase
                    && p.is_terminal()
                {
                    return Some(p);
                }
            }
            Err(_) => return None,
        }
    }
}

#[test]
#[ignore = "needs the bundled model and the private clip; run explicitly before release"]
fn the_gui_backend_path_transcribes_a_real_recording() {
    let Some(clip) = clip() else {
        println!("SKIP: scratch/clip.wav is absent");
        return;
    };
    if !bundled_model_present() {
        println!("SKIP: the bundled SraVaani model is absent");
        return;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let store = vaani_storage::Store::open(dir.path().join("ship.db")).expect("a temp store");
    let backend = vaani_app::backend::AppBackend::new(store).expect("the real backend");

    // 1. The model picker must show SraVaani as usable, not as "download me".
    let models = backend.models();
    let sravaani = models
        .iter()
        .find(|c| c.id.contains("sravaani"))
        .expect("the bundled model must appear in the picker");
    assert!(
        sravaani.available,
        "the bundled model must be selectable without a token or a download: {:?}",
        sravaani.detail
    );
    let detail = sravaani.detail.as_deref().unwrap_or("");
    assert!(
        !detail.to_lowercase().contains("token"),
        "no token may be mentioned for the bundled model: {detail}"
    );
    println!("ok  model picker: {} / {}", sravaani.id, sravaani.label);

    // 2. Enqueue, exactly as the Transcribe button does.
    let events = backend.subscribe();
    let id = backend
        .enqueue(&clip)
        .expect("enqueue must accept the clip");
    println!("ok  enqueued {}", id.as_str());

    // 3. Wait for a terminal phase, exactly as the UI's event pump does.
    let phase = wait_for_terminal(&events, &id, Duration::from_secs(600))
        .expect("the job must reach a terminal phase within 10 minutes");
    assert_eq!(
        phase,
        vaani_core::JobPhase::Complete,
        "a real recording must transcribe, not fail"
    );
    println!("ok  job reached Complete");

    // 4. The transcript the pane would render.
    let transcript = backend
        .transcript(&id)
        .expect("transcript read must not fail")
        .expect("a completed job must have a transcript");
    let words: usize = transcript.segments.iter().map(|s| s.words.len()).sum();
    assert!(
        words > 100,
        "a 14-minute meeting must have words, got {words}"
    );
    assert!(
        transcript.segments.len() < words,
        "grouping must have run: {} segments for {words} words",
        transcript.segments.len()
    );
    // Timings must be monotone and inside the media.
    let mut prev_end = vaani_core::Time::ZERO;
    for s in &transcript.segments {
        assert!(s.start <= s.end, "segment ends before it starts: {s:?}");
        assert!(
            s.start >= prev_end
                || s.start == vaani_core::Time::ZERO && prev_end == vaani_core::Time::ZERO,
            "segments out of order"
        );
        prev_end = prev_end.max(s.end);
    }
    println!(
        "ok  transcript: {} segments, {words} words, span {:?}..{:?}",
        transcript.segments.len(),
        transcript.segments.first().map(|s| s.start.as_millis()),
        transcript.segments.last().map(|s| s.end.as_millis()),
    );

    // 5. Speakers, if diarization ran.
    let speakers = backend.speakers(&id).expect("speakers read must not fail");
    println!("ok  speakers: {}", speakers.len());

    // 6. Export, exactly as the Export button does. `export()` is asynchronous:
    // it spawns a worker and returns at once, and the file lands later,
    // announced by `ExportFinished`. The UI waits for that event, so this
    // test does too — reading the directory immediately would race the
    // worker and usually lose.
    let out = dir.path().join("exports");
    std::fs::create_dir_all(&out).expect("export dir");
    for format in [ExportFormat::Text, ExportFormat::Srt, ExportFormat::Json] {
        backend
            .export(&ExportRequest {
                job: id.clone(),
                format,
                target: ExportTarget::Directory(out.clone()),
            })
            .expect("export must be accepted");
    }
    let mut finished = 0;
    let deadline = Instant::now() + Duration::from_secs(120);
    while finished < 3 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match events.recv_timeout(remaining) {
            Ok(ControllerEvent::ExportFinished { job, paths }) if job == id => {
                finished += 1;
                for p in &paths {
                    assert!(p.exists(), "exported file must exist: {}", p.display());
                }
                println!("ok  export finished: {}", paths[0].display());
            }
            Ok(ControllerEvent::ExportFailed { job, error }) if job == id => {
                panic!("export failed: {error}");
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(finished, 3, "all three exports must finish");
    let files: Vec<_> = std::fs::read_dir(&out)
        .expect("read exports")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(files.len(), 3, "three exports must exist");
    for f in &files {
        assert!(
            f.metadata().expect("stat").len() > 0,
            "export must not be empty: {}",
            f.path().display()
        );
    }
    println!("ok  3 exports written and non-empty");

    // 7. The job list the window renders on reopen.
    let jobs = backend.jobs();
    assert!(
        jobs.iter().any(|j| j.id == id),
        "the completed job must be listed"
    );
    println!("ok  job listed; SHIP GATE PASSED");
}
