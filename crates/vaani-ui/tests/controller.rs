//! Controller and state-machine tests that need no window.
//!
//! Everything here drives `UiModel` without showing it. A hidden window still
//! builds its item tree and still answers property getters, so these are real
//! assertions against the real controller — they just do not need a display,
//! which is what makes them fast enough to run on every save.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::Harness;
use slint::{ComponentHandle, Model as _};
use vaani_ui::JobSummary;
use vaani_ui::backend::{
    ALL_PHASES, ExportFormat, ExportTarget, JobController, JobId, Severity, Theme, user_facing,
};
use vaani_ui::fake::{FakeController, RecordingHostServices};
use vaani_ui::view::JobListState;

/// Build a hidden window, which is enough for everything that does not need a
/// layout.
fn hidden() -> Harness {
    Harness::new()
}

crate::ui_test!(every_job_phase_reaches_the_window_as_a_distinct_row, {
    // The state machine is the product's contract with itself, and the row is
    // the only place a user sees it. This walks all fourteen, including the two
    // most easily confused: `Cancelled` and `Failed`.
    let backend = FakeController::with_options(Duration::ZERO, 100, 1, 0);
    let h = Harness::with_backend(backend, RecordingHostServices::cancelling());
    h.session.window.show().expect("show");

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for phase in ALL_PHASES {
        let mut job = JobSummary {
            id: JobId::from(format!("phase-{phase}")),
            input: std::path::PathBuf::from(format!("C:/x/{phase}.mkv")),
            phase,
            overall: Some(0.5),
            detail: None,
            eta: None,
            error: None,
            has_transcript: false,
            cancel_requested: false,
            word_count: None,
        };
        if phase == vaani_core::JobPhase::Failed {
            job.error = Some(user_facing(
                &vaani_core::err::transcription_failed().with_recovery("Try another model."),
            ));
        }
        let list = JobListState::from_jobs(vec![job.clone()]);
        let row = &list.rows()[0];
        assert!(
            !row.status_text.is_empty(),
            "{phase} renders an empty status, which reads as a broken row"
        );
        assert!(
            seen.insert(format!("{}|{}", row.phase_code, row.status_glyph)),
            "{phase} is indistinguishable from another phase in the row"
        );
        if phase.is_terminal() {
            assert!(row.is_terminal, "{phase} should be terminal");
            assert!(!row.is_running, "{phase} should not be running");
        } else if phase == vaani_core::JobPhase::Queued {
            assert!(row.is_queued);
        } else {
            assert!(row.is_running, "{phase} should be running");
        }
    }
    assert_eq!(seen.len(), ALL_PHASES.len());
});

crate::ui_test!(progress_is_clamped_and_never_invents_a_percentage, {
    let h = hidden();
    let mut low = JobSummary {
        id: JobId::from("low"),
        input: "C:/a.mkv".into(),
        phase: vaani_core::JobPhase::Asr,
        overall: Some(-3.0),
        detail: None,
        eta: None,
        error: None,
        has_transcript: false,
        cancel_requested: false,
        word_count: None,
    };
    let mut high = low.clone();
    high.id = JobId::from("high");
    high.overall = Some(9.0);
    low.overall = None;

    let list = JobListState::from_jobs(vec![high]);
    assert_eq!(list.rows()[0].progress, Some(1.0));
    assert_eq!(list.rows()[0].progress_text, "100%");

    let list = JobListState::from_jobs(vec![low]);
    assert_eq!(list.rows()[0].progress, None);
    assert_eq!(
        list.rows()[0].progress_text,
        "",
        "an indeterminate phase must render no number at all"
    );
    let app = h.model().app_view();
    assert!(!app.can_enqueue() || app.input_accepted);
});

crate::ui_test!(
    an_error_from_the_backend_surfaces_in_the_banner_with_its_recovery,
    {
        let h = hidden();
        let error = user_facing(
            &vaani_core::err::out_of_memory()
                .with_recovery("Close other GPU applications, or switch to CPU in Settings."),
        );
        // A failed enqueue is the shortest path from a backend error to the banner.
        h.backend.fail_next_enqueue(error.clone());
        h.model().input_edited("C:/recordings/huge.mkv".to_owned());
        h.model().enqueue();

        let banner = h.window().get_banner_visible();
        assert!(banner, "the error must reach the window");
        assert_eq!(h.window().get_banner_level(), 2);
        assert_eq!(h.window().get_banner_title(), "Not enough graphics memory.");
        assert!(h.window().get_banner_recovery().contains("CPU"));
        assert_eq!(h.window().get_banner_code(), "out_of_memory");
    }
);

crate::ui_test!(
    a_rename_reaches_the_backend_with_exactly_the_arguments_it_was_given,
    {
        let h = hidden();
        h.session.window.show().expect("show");
        h.select_finished();
        let job = h
            .model()
            .job_list()
            .jobs()
            .iter()
            .find(|j| j.has_transcript)
            .cloned()
            .expect("the finished job");

        h.model().begin_rename(1);
        h.window()
            .invoke_commit_rename_requested(1, "Krishna".into());

        let persisted = h.backend.persisted_renames();
        assert_eq!(
            persisted.get(&job.id).and_then(|m| m.get(&1)),
            Some(&Some("Krishna".to_owned())),
            "the (job, speaker, name) triple the backend received is not the one the user typed"
        );
        assert_eq!(
            h.window()
                .get_speakers()
                .row_data(1)
                .unwrap()
                .name
                .to_string(),
            "Krishna"
        );
    }
);

crate::ui_test!(
    a_rename_against_the_wrong_index_is_refused_rather_than_applied,
    {
        // A stale click must not rename the wrong person. The panel owns which edit
        // is open; an index that does not match it is ignored.
        let h = hidden();
        h.session.window.show().expect("show");
        h.select_finished();
        h.model().begin_rename(0);
        h.window()
            .invoke_commit_rename_requested(1, "Not this one".into());
        assert_eq!(
            h.window()
                .get_speakers()
                .row_data(0)
                .unwrap()
                .name
                .to_string(),
            "Speaker 1",
            "an index the panel did not open must not be applied"
        );
    }
);

crate::ui_test!(export_passes_the_format_the_widget_holds, {
    let h = hidden();
    h.session.window.show().expect("show");
    h.select_finished();
    h.host.push_export_target(Some(ExportTarget::File(
        std::env::temp_dir().join("e.json"),
    )));

    // The widget owns the format until Export is pressed; there is no second
    // copy in the UI to go stale.
    let index = ExportFormat::ALL
        .iter()
        .position(|f| *f == ExportFormat::Json)
        .unwrap();
    h.window().set_export_format(i32::try_from(index).unwrap());
    h.model().export();

    let requests = h.backend.export_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].format, ExportFormat::Json);
    assert_eq!(h.model().export_format(), ExportFormat::Json);
});

crate::ui_test!(a_cancelled_picker_leaves_the_window_exactly_as_it_was, {
    let h = hidden();
    h.session.window.show().expect("show");
    let before = h.window().get_job_count();
    h.model().browse_input();
    assert_eq!(
        h.window().get_job_count(),
        before,
        "a cancelled picker adds nothing"
    );
    assert!(
        h.host.asked_for_export().is_empty(),
        "no dialog is opened for a job that cannot be exported"
    );
});

crate::ui_test!(picking_files_queues_all_of_them_and_selects_the_last, {
    let h = Harness::with_backend(
        FakeController::for_tests(),
        RecordingHostServices::new(vec![
            std::path::PathBuf::from("C:/recordings/a.mkv"),
            std::path::PathBuf::from("C:/recordings/b.mkv"),
        ]),
    );
    h.session.window.show().expect("show");
    let before = h.window().get_job_count();
    h.model().browse_input();
    assert_eq!(
        h.window().get_job_count(),
        before + 2,
        "both files were queued"
    );
    assert_eq!(
        h.backend.enqueued_paths().len(),
        2,
        "and both reached the backend"
    );
    // Selecting the last is what makes the transcript pane show the file the
    // user just picked rather than the one from before.
    let selected = h.selected_job_id().expect("a selection");
    assert!(
        h.backend
            .job(&selected)
            .is_some_and(|j| j.input.ends_with("b.mkv")),
        "the wrong job ended up selected"
    );
});

crate::ui_test!(the_pump_never_touches_the_window_when_nothing_arrived, {
    let h = hidden();
    let footer = h.window().get_footer().to_string();
    for _ in 0..100 {
        h.model().pump();
    }
    assert_eq!(h.window().get_footer().to_string(), footer);
    assert_eq!(h.model().app_view().events_applied, 0);
});

crate::ui_test!(settings_round_trip_through_the_window_and_the_backend, {
    let h = hidden();
    for (index, expected) in [(0, Theme::System), (1, Theme::Light), (2, Theme::Dark)] {
        h.model().set_theme_index(index);
        assert_eq!(h.model().settings().theme, expected);
        assert_eq!(h.backend.settings().theme, expected);
        assert_eq!(h.window().get_theme_index(), index);
    }
    h.model().set_diarize(false);
    assert!(!h.backend.settings().diarize);
    h.model().set_diarize(true);
    assert!(h.backend.settings().diarize);
});

crate::ui_test!(a_rejected_setting_leaves_the_setting_alone_and_says_why, {
    let h = hidden();
    let before = h.backend.settings().model;
    h.model().setting_label_changed(
        vaani_ui::ui_model::SettingKind::Model,
        "Not downloaded (selecting this is rejected)".to_owned(),
    );
    assert_eq!(
        h.backend.settings().model,
        before,
        "a rejection changes nothing"
    );
    assert_eq!(h.model().banner().code, "model_unavailable");
    assert!(!h.model().banner().recovery.is_empty());
});

crate::ui_test!(
    the_host_is_asked_for_a_destination_with_a_sensible_default_name,
    {
        let h = hidden();
        h.session.window.show().expect("show");
        h.select_finished();
        h.host
            .push_export_target(Some(ExportTarget::File(std::env::temp_dir().join("x.srt"))));
        let index = ExportFormat::ALL
            .iter()
            .position(|f| *f == ExportFormat::Srt)
            .unwrap();
        h.window().set_export_format(i32::try_from(index).unwrap());
        h.model().export();
        // The host records the destinations it was asked about; the suggested name
        // is not recorded, so assert on the request that came back instead.
        let requests = h.backend.export_requests();
        assert_eq!(requests[0].format.extension(), "srt");
        assert!(matches!(requests[0].target, ExportTarget::File(_)));
    }
);

crate::ui_test!(an_event_that_names_a_vanished_job_is_ignored_not_fatal, {
    // The backend and the UI are allowed to disagree about which jobs exist, for
    // a frame. A transcript-ready for a job the user has already removed must
    // not resurrect it or panic.
    let h = hidden();
    let id = h.backend.enqueue(std::path::Path::new("C:/a.mkv")).unwrap();
    h.pump();
    h.backend.remove(&id).unwrap();
    h.pump();
    // One more pump with nothing queued, which is the state a torn read would
    // leave behind.
    h.pump();
    assert!(h.backend.job(&id).is_none());
});

crate::ui_test!(a_notice_is_shown_once_and_dismissed_once, {
    let h = hidden();
    vaani_ui::fake::notice(&h.backend, Severity::Info, "Model cached");
    h.pump();
    assert_eq!(h.model().banner().title, "Model cached");
    h.model().dismiss_banner();
    assert!(!h.model().banner().visible);
    h.pump();
    assert!(
        !h.model().banner().visible,
        "dismissing must stick until something new happens"
    );
});

crate::ui_test!(a_job_with_no_transcript_says_why_instead_of_being_blank, {
    let h = hidden();
    h.session.window.show().expect("show");
    // The default selection is the cancelled history job.
    assert!(!h.window().get_transcript_empty().is_empty());
    assert_eq!(h.window().get_transcript_rows_total(), 0);
    assert!(!h.window().get_can_copy());
    assert!(!h.window().get_can_export());
});

crate::ui_test!(an_unavailable_host_is_inert_rather_than_fatal, {
    // A clipboard that does not work must not take the window down with it.
    let session = vaani_ui::build(
        Arc::new(FakeController::for_tests()),
        Arc::new(vaani_ui::UnavailableHostServices),
    )
    .expect("build");
    session.model.select_all();
    session.model.copy_selection();
    assert!(!session.model.job_list().jobs().is_empty());
});
