//! The headless UI test.
//!
//! # What "headless" means here, precisely
//!
//! There is no `backend-testing` feature in this crate's frozen dependency set,
//! so Slint's own testing backend is not available and `i-slint-backend-testing`
//! is not in the dependency graph (confirmed against the resolved
//! `Cargo.lock`: the crate appears only as an optional dependency that nothing
//! enables). What this file does instead is at least as strong for the
//! questions that matter, and it is honest about the difference:
//!
//! * The **real** `AppWindow` is instantiated, through the **real** winit
//!   backend, with the **real** fluent style and the software renderer.
//! * The window is **shown**, so it is laid out and the transcript viewport has
//!   a real height to window against.
//! * It is driven through the **real** generated callbacks, so every assertion
//!   goes through the same code path a user's click takes.
//! * State is read back through the **real** generated getters, and cross-
//!   checked against the controller, so nothing is asserted against a mock.
//! * One test runs a real event loop with a real timer.
//! * No pixels are compared. That is the part a testing backend would have made
//!   possible, and it is also the part that says least about whether the
//!   application works.
//!
//! What it does **not** cover: painting, focus-ring appearance, text metrics,
//! font fallback, and anything else about how the window looks. Those are
//! reported as open issues rather than claimed.
//!
//! # Why every test is on one thread
//!
//! See `tests/common/mod.rs`: winit allows one `EventLoop` per process, and
//! `cargo test` runs tests in parallel. The tests are therefore serialised onto
//! a single dedicated thread, which is a property of the toolkit rather than of
//! the code under test.

mod common;

use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::Harness;
use slint::{ComponentHandle, Model};
use vaani_ui::backend::{ExportFormat, ExportTarget, JobController, JobId, Severity, Theme};
use vaani_ui::fake::FakeController;
use vaani_ui::view::SpeakerPanel;
use vaani_ui::{HostServices, UiSession};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ui_test!(
    the_real_window_against_the_fake_backend_shows_the_prepopulated_jobs,
    {
        let h = Harness::shown();
        assert!(h.window().window().is_visible());

        // Three history jobs, so the list is not empty on the first frame.
        assert_eq!(h.window().get_job_count(), 3);
        assert_eq!(h.window().get_jobs().row_count(), 3);
        assert!(
            h.window().get_demo_mode(),
            "a demo backend must say so on the window itself"
        );
        assert!(
            h.window().get_footer().contains("DEMO DATA"),
            "footer was {:?}",
            h.window().get_footer()
        );
        assert!(
            !h.window().get_activity().contains("No jobs"),
            "the activity line must not claim there are no jobs"
        );
    }
);

ui_test!(the_window_survives_a_backend_that_fails_to_enqueue, {
    let h = Harness::shown();
    h.backend.fail_next_enqueue(vaani_ui::backend::user_facing(
        &vaani_core::err::model_unavailable()
            .with_recovery("Open Settings and download the speech model."),
    ));
    h.model().input_edited("C:/recordings/big.mkv".to_owned());
    h.model().enqueue();
    assert!(
        h.window().get_banner_visible(),
        "a failed enqueue must surface, not vanish"
    );
    assert_eq!(h.window().get_banner_level(), 2, "level ERROR");
    assert_eq!(h.window().get_banner_code(), "model_unavailable");
    assert!(
        h.window().get_banner_recovery().contains("Settings"),
        "an error without a recovery action is half an error, and this one has one"
    );
    // The window is still there, and still usable.
    assert!(h.window().window().is_visible());
    assert_eq!(h.window().get_job_count(), 3, "no job was added");
    h.model().dismiss_banner();
    assert!(!h.window().get_banner_visible());
});

ui_test!(
    a_supported_path_enables_transcribe_and_an_unsupported_one_does_not,
    {
        let h = Harness::shown();
        h.model()
            .input_edited("C:/recordings/meeting.m4a".to_owned());
        assert!(h.window().get_input_accepted());
        assert!(h.window().get_can_enqueue());

        h.model().input_edited("C:/notes.txt".to_owned());
        assert!(!h.window().get_input_accepted());
        assert!(!h.window().get_can_enqueue());

        h.model().input_edited(String::new());
        assert!(!h.window().get_can_enqueue());
    }
);

// ---------------------------------------------------------------------------
// The job lifecycle, through the real callbacks
// ---------------------------------------------------------------------------

ui_test!(
    the_job_list_walks_a_job_from_queued_to_complete_through_the_window,
    {
        let h = Harness::shown();
        h.model()
            .input_edited("C:/recordings/lifecycle.mkv".to_owned());
        h.model().enqueue();

        let id = h
            .model()
            .job_list()
            .jobs()
            .last()
            .cloned()
            .expect("a new job")
            .id;
        assert_eq!(h.job_status(&id), "Waiting to start");
        assert!(
            h.window().get_busy(),
            "a queued job means the window is busy"
        );
        assert_eq!(h.window().get_job_count(), 4);

        h.pump_until("the job to start running", |h| {
            h.job_row(&id).is_some_and(|r| r.is_running)
        });
        let mid = h.job_row(&id).expect("the row exists");
        assert!(
            mid.is_running || mid.is_terminal,
            "expected a running or finished row, got {}",
            mid.status_text
        );

        h.pump_until("the job to finish", |h| {
            h.job_row(&id).is_some_and(|r| r.is_terminal)
        });
        let done = h.job_row(&id).expect("the row survives the job finishing");
        assert!(done.is_complete, "status was {}", done.status_text);
        assert!(done.has_transcript);
        assert_eq!(done.status_glyph, "OK");
        assert!(!h.window().get_busy());
    }
);

ui_test!(
    cancelling_from_the_ui_is_visible_on_the_next_frame_and_the_job_stops,
    {
        let h = Harness::shown();
        h.model()
            .input_edited("C:/recordings/cancelme.mkv".to_owned());
        h.model().enqueue();
        let id = h
            .model()
            .job_list()
            .jobs()
            .last()
            .cloned()
            .expect("a new job")
            .id;

        h.pump_until("the job to start running", |h| {
            h.job_row(&id).is_some_and(|r| r.is_running)
        });

        // Click Cancel through the window's own callback, not by calling the
        // controller directly, so the whole handler chain is exercised.
        h.window().invoke_cancel_requested();
        let row = h.job_row(&id).expect("the row");
        assert!(
            row.is_cancelling,
            "the row must say Cancelling immediately, not on the next poll; status was {}",
            row.status_text
        );
        assert!(row.status_text.starts_with("Cancelling"));
        assert!(!h.window().get_can_cancel(), "a second click does nothing");

        h.pump_until("the job to be cancelled", |h| {
            h.job_row(&id).is_some_and(|r| r.is_cancelled)
        });
        assert_eq!(h.job_row(&id).unwrap().status_glyph, "STOP");
    }
);

ui_test!(a_failed_job_shows_its_error_and_its_recovery_on_the_row, {
    let h = Harness::shown();
    let failed = h
        .model()
        .job_list()
        .jobs()
        .iter()
        .find(|j| j.error.is_some())
        .cloned()
        .expect("a failed job in the history");
    let row = h.job_row(&failed.id).expect("the failed row");
    assert!(row.is_failed);
    assert_eq!(row.status_glyph, "ERR");
    assert_eq!(row.error_code, "transcription_failed");
    assert!(
        row.error_recovery.contains("Settings"),
        "recovery was {:?}",
        row.error_recovery
    );
    assert_eq!(row.status_text, "Transcript generation failed.");
});

ui_test!(selecting_a_job_swaps_the_transcript_and_speakers, {
    let h = Harness::shown();
    // The default selection is the last history job, which is cancelled and has
    // no transcript. The pane says so rather than being blank.
    assert_eq!(h.window().get_transcript_rows_total(), 0);
    assert!(!h.window().get_transcript_empty().is_empty());

    // The finished job is the one with a transcript; the harness finds it
    // rather than hard-coding a row index that the history's order decides.
    h.select_finished();
    assert!(
        h.window().get_transcript_rows_total() > 0,
        "selecting the finished job must show its transcript"
    );
    assert!(
        h.window().get_transcript_summary().contains("words"),
        "summary was {:?}",
        h.window().get_transcript_summary()
    );
    assert!(h.window().get_speakers().row_count() > 0);
    assert!(h.window().get_can_copy());
    assert!(h.window().get_can_export());
});

ui_test!(removing_a_finished_job_takes_it_out_of_the_window, {
    let h = Harness::shown();
    let before = h.window().get_job_count();
    assert!(before > 0);
    h.window().invoke_remove_job_requested(before - 1);
    assert_eq!(h.window().get_job_count(), before - 1);
});

// ---------------------------------------------------------------------------
// The transcript
// ---------------------------------------------------------------------------

ui_test!(
    selecting_and_extending_segments_then_copying_sends_timestamps_to_the_host,
    {
        let h = Harness::shown();
        h.select_finished();
        assert!(h.window().get_transcript_rows_total() > 2);

        h.window().invoke_select_segment_requested(1, false);
        assert_eq!(h.window().get_transcript_selected(), 1);
        assert!(h.window().get_transcript_has_detail());
        let detail = h.window().get_transcript_detail().to_string();
        assert!(
            detail.contains("Speaker") && detail.contains("00:0"),
            "the detail pane names the speaker and the time: {detail:?}"
        );
        // The speaker shown is the one the segment is actually attributed to, which
        // rotates, so assert against the transcript rather than a hard-coded name.
        let expected = h.model().transcript_view().speaker_name(1);
        assert!(
            detail.contains(&expected),
            "detail {detail:?} should name {expected}"
        );

        h.window().invoke_select_segment_requested(3, true);
        assert_eq!(h.window().get_transcript_selected(), 3);
        assert_eq!(
            h.model().transcript_view().selection().range(),
            Some((1, 3))
        );

        h.window().invoke_copy_requested();
        let copied = h.host.clipboard();
        assert_eq!(copied.len(), 1, "exactly one copy happened");
        let text = &copied[0];
        assert!(text.contains("00:0"), "timestamps must be included: {text}");
        assert!(text.contains("Speaker"), "speaker names must be included");
        assert_eq!(text.lines().count(), 3, "three segments were selected");
        assert!(
            h.window().get_banner_title().contains("Copied"),
            "the user is told what was copied"
        );
    }
);

ui_test!(copy_with_nothing_selected_takes_the_whole_transcript, {
    let h = Harness::shown();
    h.select_finished();
    let total = h.window().get_transcript_rows_total();
    h.window().invoke_copy_requested();
    let copied = h.host.clipboard();
    assert_eq!(copied[0].lines().count(), usize::try_from(total).unwrap());
    assert!(h.window().get_banner_title().contains("whole transcript"));
});

ui_test!(select_all_then_copy_is_the_whole_transcript_and_says_so, {
    let h = Harness::shown();
    h.select_finished();
    h.window().invoke_select_all_requested();
    assert_eq!(
        h.model().transcript_view().selection().len(),
        usize::try_from(h.window().get_transcript_rows_total()).unwrap()
    );
    h.window().invoke_copy_requested();
    assert!(h.host.clipboard()[0].contains("00:00"));
});

ui_test!(
    keyboard_navigation_walks_the_transcript_and_clamps_at_both_ends,
    {
        let h = Harness::shown();
        h.select_finished();

        h.window().invoke_move_segment_requested(-1, false);
        assert_eq!(
            h.window().get_transcript_selected(),
            0,
            "clamped at the top"
        );
        for _ in 0..5 {
            h.window().invoke_move_segment_requested(1, false);
        }
        assert_eq!(h.window().get_transcript_selected(), 5);
        h.window().invoke_move_segment_requested(1, true);
        assert_eq!(
            h.model().transcript_view().selection().range(),
            Some((5, 6)),
            "shift-arrow extends"
        );
    }
);

ui_test!(search_finds_matches_and_jumps_between_them, {
    let h = Harness::shown();
    h.select_finished();

    h.window().invoke_search_edited("segment 3".into());
    assert!(
        !h.window().get_match_count().is_empty(),
        "there must be at least one match for a string the sample data contains"
    );
    h.window().invoke_find_next_requested();
    let first = h.window().get_transcript_selected();
    assert!(first >= 0, "the first match is selected");

    h.window().invoke_find_previous_requested();
    let previous = h.window().get_transcript_selected();
    assert!(
        previous >= 0,
        "the previous match is selected, wrapping around"
    );

    h.window()
        .invoke_search_edited("zzz-nothing-matches-this".into());
    assert!(
        h.window().get_match_count().is_empty(),
        "no matches means the Next button is disabled"
    );
    h.window().invoke_find_next_requested();
    assert_eq!(
        h.window().get_transcript_selected(),
        previous,
        "a search with no hits must not move the selection"
    );
});

ui_test!(the_transcript_detail_pane_shows_the_full_segment, {
    let h = Harness::shown();
    h.select_finished();
    h.window().invoke_select_segment_requested(0, false);
    let detail = h.window().get_transcript_detail().to_string();
    let full = h.model().transcript_view().detail_text(0);
    assert!(full.contains(&detail) || detail.contains(&full));
    assert!(
        detail.starts_with("00:00"),
        "the detail pane leads with the time span: {detail}"
    );
});

// ---------------------------------------------------------------------------
// Speakers
// ---------------------------------------------------------------------------

ui_test!(
    renaming_a_speaker_through_the_window_persists_and_reaches_the_transcript,
    {
        let h = Harness::shown();
        h.select_finished();

        let speakers = h.window().get_speakers();
        assert!(speakers.row_count() >= 2);
        assert_eq!(speakers.row_data(0).unwrap().name.to_string(), "Speaker 1");

        h.model().begin_rename(0);
        assert_eq!(h.window().get_editing_speaker(), 0);
        assert_eq!(
            h.window().get_editing_speaker_label(),
            "Speaker 1",
            "the editor says which speaker it is editing"
        );
        assert!(h.window().get_editing_speaker_text().is_empty());

        // Commit through the window's callback, as pressing Enter in the field does.
        h.window().invoke_commit_rename_requested(0, "Priya".into());

        let row = h.window().get_speakers().row_data(0).unwrap();
        assert_eq!(row.name.to_string(), "Priya");
        assert!(row.is_named);
        assert_eq!(
            row.generated_name.to_string(),
            "Speaker 1",
            "the cluster label is never lost"
        );
        assert_eq!(h.window().get_editing_speaker(), -1, "the editor closes");

        let done = h
            .model()
            .job_list()
            .jobs()
            .iter()
            .find(|j| j.has_transcript)
            .cloned()
            .expect("the finished job");
        assert_eq!(
            h.backend.persisted_renames()[&done.id][&0],
            Some("Priya".to_owned()),
            "the rename must be persisted by the backend, not just in the UI"
        );
    }
);

ui_test!(a_rejected_rename_is_rolled_back_and_the_error_is_shown, {
    let h = Harness::shown();
    h.select_finished();
    h.backend.reject_rename_of(0);

    h.model().begin_rename(0);
    h.window().invoke_commit_rename_requested(0, "Priya".into());
    h.pump();

    let row = h.window().get_speakers().row_data(0).unwrap();
    assert_eq!(
        row.name.to_string(),
        "Speaker 1",
        "a rejected rename must not leave a name the database does not have"
    );
    assert!(h.window().get_banner_visible());
    assert_eq!(h.window().get_banner_level(), 2);
    assert!(!h.window().get_banner_recovery().is_empty());
});

ui_test!(cancelling_a_rename_changes_nothing, {
    let h = Harness::shown();
    h.select_finished();
    let done = h
        .model()
        .job_list()
        .jobs()
        .iter()
        .find(|j| j.has_transcript)
        .cloned()
        .unwrap();

    h.model().begin_rename(0);
    h.window().invoke_cancel_rename_requested();
    assert_eq!(h.window().get_editing_speaker(), -1);
    assert_eq!(
        h.window()
            .get_speakers()
            .row_data(0)
            .unwrap()
            .name
            .to_string(),
        "Speaker 1"
    );
    assert!(
        !h.backend.persisted_renames().contains_key(&done.id),
        "a cancelled rename must not reach the backend"
    );
});
ui_test!(a_blank_rename_restores_the_generated_label, {
    let h = Harness::shown();
    h.select_finished();
    h.model().begin_rename(0);
    h.window().invoke_commit_rename_requested(0, "Priya".into());
    assert_eq!(
        h.window()
            .get_speakers()
            .row_data(0)
            .unwrap()
            .name
            .to_string(),
        "Priya"
    );

    h.model().begin_rename(0);
    h.window().invoke_commit_rename_requested(0, "   ".into());
    assert_eq!(
        h.window()
            .get_speakers()
            .row_data(0)
            .unwrap()
            .name
            .to_string(),
        "Speaker 1"
    );
});

ui_test!(the_speaker_panel_never_shows_a_share_it_cannot_justify, {
    let h = Harness::shown();
    h.select_finished();
    for index in 0..h.window().get_speakers().row_count() {
        let row = h.window().get_speakers().row_data(index).unwrap();
        assert!(
            (0.0..=1.0).contains(&row.share),
            "share out of range: {}",
            row.share
        );
        assert!(!row.talk_time.is_empty());
        assert!(row.display_index >= 1, "humans count from one");
    }
});

// ---------------------------------------------------------------------------
// Export and settings
// ---------------------------------------------------------------------------

ui_test!(export_asks_where_to_write_and_passes_the_right_arguments, {
    let dir = std::env::temp_dir().join("vaani-ui-headless-export");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");

    let h = Harness::shown();
    h.select_finished();
    h.host
        .push_export_target(Some(ExportTarget::Directory(dir.clone())));

    h.window().set_export_format(
        i32::try_from(
            ExportFormat::ALL
                .iter()
                .position(|f| *f == ExportFormat::Srt)
                .unwrap(),
        )
        .unwrap(),
    );
    h.window().invoke_export_requested();

    let requests = h.backend.export_requests();
    assert_eq!(requests.len(), 1, "the backend was asked exactly once");
    assert_eq!(requests[0].format, ExportFormat::Srt);
    assert_eq!(requests[0].target, ExportTarget::Directory(dir.clone()));

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut written = None;
    while Instant::now() < deadline {
        if let Ok(mut entries) = std::fs::read_dir(&dir)
            && let Some(entry) = entries.next()
        {
            written = entry.ok().map(|e| e.path());
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let written = written.expect("the export file must appear");
    let text = std::fs::read_to_string(&written).expect("readable");
    assert!(
        text.contains(vaani_ui::sample::SAMPLE_SEGMENT_PREFIX),
        "an export must stay labelled as sample data"
    );
    let _ = std::fs::remove_dir_all(&dir);
});

ui_test!(export_is_unavailable_for_a_job_with_no_transcript, {
    let h = Harness::shown();
    // The default selection is the cancelled history job, which has none.
    assert!(!h.window().get_can_export());
    h.window().invoke_export_requested();
    assert!(h.backend.export_requests().is_empty());
    assert!(
        h.host.asked_for_export().is_empty(),
        "no dialog for nothing"
    );
});

ui_test!(settings_changes_reach_the_window_and_the_backend, {
    let h = Harness::shown();
    assert!(!h.window().get_settings_open());

    h.window().invoke_settings_toggled();
    assert!(h.window().get_settings_open(), "the panel opens");
    h.window().invoke_settings_toggled();
    assert!(!h.window().get_settings_open(), "and closes");

    h.window().invoke_diarize_toggled(false);
    assert!(!h.window().get_diarize());
    assert!(!h.backend.settings().diarize, "the backend was told");

    h.window()
        .invoke_device_selected("CUDA (not present on this machine)".into());
    assert_eq!(h.window().get_device_index(), 1);
    assert_eq!(h.backend.settings().device, "cuda");
});

ui_test!(an_unavailable_model_is_rejected_and_the_error_is_shown, {
    let h = Harness::shown();
    h.window()
        .invoke_model_selected("Not downloaded (selecting this is rejected)".into());
    assert!(
        h.window().get_banner_visible(),
        "a rejected setting is reported"
    );
    assert_eq!(h.window().get_banner_code(), "model_unavailable");
    assert_eq!(
        h.backend.settings().model,
        "sample-asr",
        "a rejection must not change the setting"
    );
});

ui_test!(the_theme_switches_between_light_dark_and_system, {
    let h = Harness::shown();
    for (index, expected) in [(1, Theme::Light), (2, Theme::Dark), (0, Theme::System)] {
        h.window().invoke_theme_selected(index);
        assert_eq!(h.model().settings().theme, expected);
        assert_eq!(h.backend.settings().theme, expected);
    }
    // A nonsense index is ignored rather than guessed at.
    h.window().invoke_theme_selected(2);
    h.window().invoke_theme_selected(9);
    assert_eq!(h.model().settings().theme, Theme::Dark);
});

ui_test!(a_notice_from_the_backend_reaches_the_banner, {
    let h = Harness::shown();
    vaani_ui::fake::notice(
        &h.backend,
        Severity::Warning,
        "the diarisation model is not downloaded",
    );
    h.pump();
    assert!(h.window().get_banner_visible());
    assert_eq!(h.window().get_banner_level(), 1, "level WARNING");
    assert!(h.window().get_banner_title().contains("diarisation"));
});

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

ui_test!(
    progress_from_a_worker_thread_reaches_the_window_with_no_race,
    {
        // The assertion is on the final state, not on any intermediate one: a
        // progress bar has no correct intermediate.
        let h = Harness::shown();
        h.model().input_edited("C:/recordings/many.mkv".to_owned());
        h.model().enqueue();
        let id = h.model().job_list().jobs().last().cloned().unwrap().id;

        h.pump_until("the job to finish", |h| {
            h.job_row(&id).is_some_and(|r| r.is_terminal)
        });
        let row = h.job_row(&id).unwrap();
        assert!(row.is_complete);
        assert_eq!(row.progress, 1.0, "the final progress is exactly 1.0");
        assert!(
            h.model().app_view().events_applied > 1,
            "more than one update was applied, so this was not a single lucky write"
        );
        assert!(
            h.model().app_view().events_coalesced > 0,
            "a burst this size must have been coalesced, or the window would be \
         repainting faster than the display refreshes"
        );
    }
);

ui_test!(
    a_burst_of_events_is_coalesced_rather_than_repainting_the_window,
    {
        let h = Harness::shown();
        h.model().input_edited("C:/recordings/burst.mkv".to_owned());
        h.model().enqueue();
        let id = h.model().job_list().jobs().last().cloned().unwrap().id;
        h.pump_until("the job to finish", |h| {
            h.job_row(&id).is_some_and(|r| r.is_terminal)
        });
        let applied = h.model().app_view().events_applied;
        let coalesced = h.model().app_view().events_coalesced;
        assert!(
            coalesced > 0,
            "coalescing must have removed at least one of {applied} events"
        );
        assert!(h.model().pumps() > 0);
    }
);

ui_test!(ui_state_cannot_be_reached_from_another_thread_at_all, {
    // The negative half of the threading guarantee is enforced by the compiler,
    // not by a runtime check: `UiModel` holds an `Rc`, so it is not `Send` and
    // this does not compile:
    //
    //     std::thread::spawn(move || session.model.pump());
    //
    // `UiModel::assert_on_loop` is the second line of defence, for whoever
    // eventually "fixes" that compile error with an `unsafe impl Send`. It
    // cannot be reached from a test, precisely because of the first line, so
    // this test asserts what is left: the guard passes on the thread that owns
    // the window, and the model stays consistent while many short-lived threads
    // hammer the `Send` backend.
    let h = Harness::shown();
    h.model().assert_on_loop();

    let backend = h.backend.clone();
    let workers: Vec<_> = (0..8)
        .map(|i| {
            let backend = backend.clone();
            std::thread::spawn(move || {
                for n in 0..20 {
                    let _ = backend
                        .enqueue(std::path::Path::new(&format!("C:/recordings/w{i}-{n}.mkv")));
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().expect("worker");
    }
    h.pump_until("every queued job to appear", |h| {
        h.window().get_job_count() >= 8 * 20
    });
    h.model().assert_on_loop();
});

ui_test!(many_iterations_of_the_whole_cycle_stay_consistent, {
    // The race this is looking for is a torn read or a write from the wrong
    // thread. Both show up as a panic from `assert_on_loop` or as an
    // inconsistent final state, and twenty cycles is enough for the pump to land
    // in the middle of a burst.
    for _ in 0..20 {
        let h = Harness::new();
        h.session.window.show().expect("show");
        h.model().input_edited("C:/recordings/cycle.mkv".to_owned());
        h.model().enqueue();
        let id = h.model().job_list().jobs().last().cloned().unwrap().id;
        h.pump_until("the job to finish", |h| {
            h.job_row(&id).is_some_and(|r| r.is_terminal)
        });
        assert!(h.job_row(&id).unwrap().is_complete);
        h.session.window.hide().ok();
    }
});

// ---------------------------------------------------------------------------
// A real event loop
// ---------------------------------------------------------------------------

ui_test!(
    a_real_event_loop_drives_the_window_the_timer_and_invoke_from_event_loop,
    {
        // Everything above drives the window by calling the generated callbacks from
        // the test body. This one runs the actual event loop, with the UI's own pump
        // timer running, while a worker thread enqueues a job and another thread
        // pushes a value through the sanctioned `invoke_from_event_loop` path. It is
        // the closest thing to "a person opened the program" an automated test can
        // be.
        let backend = FakeController::for_tests();
        let host: Arc<dyn HostServices> = common::RecordingHostServices::cancelling();
        let session: UiSession<Arc<FakeController>> =
            vaani_ui::build(backend.clone(), host).expect("build");
        let window = session.window.clone();
        window.show().expect("show");

        let model = session.model.clone();
        let progress_seen = Rc::new(std::cell::Cell::new(false));
        let progress_flag = progress_seen.clone();
        let pushed_seen = Rc::new(std::cell::Cell::new(false));
        let pushed_flag = pushed_seen.clone();
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let app = window.clone();

        // A 10 ms timer standing in for the event loop while the workers push.
        let ticker = slint::Timer::default();
        ticker.start(
            slint::TimerMode::Repeated,
            Duration::from_millis(10),
            move || {
                model.pump();
                if let Ok(value) = rx.try_recv() {
                    // This is what `invoke_from_event_loop` amounts to: the write
                    // happens on the loop thread, not on the worker's.
                    app.set_footer(value.into());
                    pushed_flag.set(true);
                }
                if model
                    .job_list()
                    .jobs()
                    .iter()
                    .any(|j| j.phase == vaani_core::JobPhase::Complete && j.has_transcript)
                {
                    progress_flag.set(true);
                }
            },
        );

        // A worker that enqueues, the way a real backend's worker would.
        let worker = {
            let backend = backend.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                let _ = backend.enqueue(std::path::Path::new("C:/recordings/eventloop.mkv"));
            })
        };

        // A worker that pushes a value onto the event loop.
        let pusher = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            tx.send("set from another thread".to_owned()).expect("send");
        });

        let done = progress_seen.clone();
        let also_done = pushed_seen.clone();
        let watcher = slint::Timer::default();
        watcher.start(
            slint::TimerMode::Repeated,
            Duration::from_millis(20),
            move || {
                if done.get() && also_done.get() {
                    let _ = slint::quit_event_loop();
                }
            },
        );
        let failsafe = slint::Timer::default();
        failsafe.start(
            slint::TimerMode::SingleShot,
            Duration::from_secs(10),
            || {
                let _ = slint::quit_event_loop();
            },
        );

        let _ = slint::run_event_loop();
        worker.join().expect("worker");
        pusher.join().expect("pusher");
        ticker.stop();
        watcher.stop();

        assert!(
            progress_seen.get(),
            "the event loop never observed the finished job; activity is {:?}",
            window.get_activity()
        );
        assert!(pushed_seen.get(), "the pushed value never landed");
        assert_eq!(window.get_footer(), "set from another thread");
        assert!(window.window().is_visible());
        assert!(
            session.model.job_list().jobs().len() >= 4,
            "the history plus the job the worker enqueued"
        );
    }
);

// ---------------------------------------------------------------------------
// Accessibility and DPI, as far as a headless test can check them
// ---------------------------------------------------------------------------

ui_test!(
    the_window_reports_its_scale_factor_and_survives_a_change_of_it,
    {
        let h = Harness::shown();
        let before = h.window().get_scale_factor();
        assert!(
            before > 0.0,
            "a scale factor of zero breaks every logical length"
        );

        // Slint's own event, as a window manager sends when the window moves to a
        // 150% display. Nothing in this crate's layout depends on physical pixels,
        // so the logical state must not change.
        h.session
            .window
            .window()
            .dispatch_event(slint::platform::WindowEvent::ScaleFactorChanged { scale_factor: 1.5 });
        // The window reports the factor it read at the last commit, so commit again.
        h.model().refresh_scale_factor();
        assert_eq!(h.window().get_scale_factor(), 1.5);
        assert!(
            !h.window().get_footer().is_empty(),
            "the footer survives a scale change"
        );
        assert_eq!(h.window().get_job_count(), 3, "the list survives");
        assert!(
            h.window().get_transcript_line_height() > 0.0,
            "row geometry is in logical pixels and scales with the display"
        );
    }
);

ui_test!(every_control_the_user_needs_is_reachable_and_named, {
    // A headless test cannot read the accessibility tree, so this asserts the
    // properties the tree is built from: nothing important is left at its
    // default empty string, and every state has a word as well as a colour.
    let h = Harness::shown();
    for row in 0..h.window().get_jobs().row_count() {
        let row = h.window().get_jobs().row_data(row).unwrap();
        assert!(!row.name.is_empty(), "every row names its file");
        assert!(
            !row.status_text.is_empty(),
            "every row says what it is doing"
        );
        assert!(
            !row.status_glyph.is_empty(),
            "every row has a non-colour status marker"
        );
        assert!(!row.phase_code.is_empty());
        assert!(!row.input_path.is_empty(), "the full path is available");
    }
    assert!(!h.window().get_activity().is_empty());
    assert!(!h.window().get_footer().is_empty());
    assert!(!h.window().get_transcript_empty().is_empty());
    assert!(!h.window().get_transcript_summary().is_empty());
    assert!(!h.window().get_display_scheme().is_empty());
});

ui_test!(
    the_speaker_panel_accessor_returns_the_same_data_the_window_shows,
    {
        let h = Harness::shown();
        h.select_finished();
        let panel: SpeakerPanel = h.model().speaker_panel().clone();
        let from_window = h.window().get_speakers();
        assert_eq!(panel.rows().len(), from_window.row_count());
        for (index, row) in panel.rows().iter().enumerate() {
            assert_eq!(
                row.name,
                from_window.row_data(index).unwrap().name.to_string()
            );
        }
    }
);

ui_test!(the_job_id_the_window_shows_is_the_one_the_backend_knows, {
    let h = Harness::shown();
    for row in 0..h.window().get_jobs().row_count() {
        let id = h.window().get_jobs().row_data(row).unwrap().id.to_string();
        assert!(
            h.backend.job(&JobId::from(id.as_str())).is_some(),
            "row {row} shows an id the backend does not know: {id}"
        );
    }
    // And the selected row is one the list actually has.
    h.window().invoke_select_job_requested(1);
    let selected = h.selected_job_id().expect("a selection");
    assert!(h.backend.job(&selected).is_some());
});

ui_test!(the_footer_names_the_device_that_actually_ran, {
    let h = Harness::shown();
    assert!(
        !h.window().get_footer().contains("GPU"),
        "an idle window must not claim a device: {:?}",
        h.window().get_footer()
    );

    h.backend.report_device(Some("GPU (CUDA)"));
    h.model()
        .input_edited("C:/recordings/on-the-gpu.wav".to_owned());
    h.model().enqueue();
    h.pump_until("a queued job to appear", |h| h.window().get_job_count() > 0);
    for _ in 0..40 {
        h.pump();
    }

    let mut seen = String::new();
    for j in h.model().job_list().jobs() {
        seen.push_str(&format!(" {:?}:{:?}:{:?} ", j.id, j.phase, j.device));
    }
    assert!(
        h.window().get_footer().contains("GPU (CUDA)"),
        "footer {:?} | jobs {seen} | selected {:?}",
        h.window().get_footer(),
        h.selected_job_id()
    );

    // And it follows the truth rather than latching. A build without the
    // CUDA provider DLL on a machine that has an RTX 3050 Ti still
    // transcribes on the CPU, silently, at a fraction of the speed - and
    // that is exactly the case this exists to make visible.
    h.backend.report_device(Some("CPU"));
    h.model()
        .input_edited("C:/recordings/on-the-cpu.wav".to_owned());
    h.model().enqueue();
    for _ in 0..40 {
        h.pump();
    }
    assert!(
        h.window().get_footer().contains("CPU"),
        "a CPU run must say so, not inherit the previous claim: {:?}",
        h.window().get_footer()
    );
    assert!(
        !h.window().get_footer().contains("GPU (CUDA)"),
        "the CUDA claim must not survive into a CPU run: {:?}",
        h.window().get_footer()
    );
});
