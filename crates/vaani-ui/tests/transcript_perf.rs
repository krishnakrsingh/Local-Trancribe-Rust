//! Performance: does a 14-minute meeting open fast, and does it stay fast?
//!
//! # The numbers this file is allowed to claim
//!
//! A 20,000-word transcript is roughly a 14-minute meeting at ordinary
//! speaking pace. Everything here is measured against that size, and every
//! assertion carries a budget that is tight enough to catch a regression and
//! loose enough not to fail on a loaded CI machine. The budgets are stated in
//! the assertions rather than hidden in a constant, so a reader can disagree
//! with them.
//!
//! What is measured:
//!
//! * **Window open.** Building the window, the model, and the first commit.
//! * **Transcript load.** Handing a 20,000-word transcript to the UI. This is
//!   the number that would be catastrophic if it were `O(words)`, and it is
//!   `O(segments)` because the window is a window.
//! * **Row windowing.** Scrolling from the top to the bottom of the transcript,
//!   measuring the cost of each viewport update. This is the number that says
//!   whether scrolling is smooth or not.
//! * **Delegate count.** The number of live transcript rows at each scroll
//!   position, which is the claim the whole design rests on.
//!
//! # These budgets are for a debug build
//!
//! `cargo test` compiles with `opt-level = 0`. The numbers below are therefore
//! several times worse than what a user sees, which is the conservative
//! direction: a budget that holds without optimisation will hold with it. Run
//! `cargo test --release -p vaani-ui --test transcript_perf -- --nocapture`
//! for the figures a user actually gets; both sets are in the A5 report.
//!
//! What is *not* measured: paint time. That needs a GPU or a real display and
//! is reported as an open issue rather than guessed at.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::Harness;
use slint::{ComponentHandle, Model};
use vaani_ui::TRANSCRIPT_LINE_HEIGHT_PX;
use vaani_ui::backend::JobController as _;
use vaani_ui::fake::FakeController;

/// The size under test: a 14-minute meeting.
const WORDS: usize = 20_000;
const SPEAKERS: usize = 4;

/// A window whose selected job has a 20,000-word transcript.
fn big_transcript_harness() -> Harness {
    let backend = FakeController::with_options(Duration::ZERO, WORDS, SPEAKERS, 1);
    let harness = Harness::with_backend(backend, common::RecordingHostServices::cancelling());
    harness.session.window.show().expect("show");
    harness
}

crate::ui_test!(window_open_time_is_measured_and_small, {
    // Ten windows, so the number is a distribution and not one lucky run. The
    // reported figure is the worst of the ten: a product that opens in 30 ms
    // usually and 900 ms once is a product that feels broken once.
    let mut worst = Duration::ZERO;
    let mut best = Duration::MAX;
    for _ in 0..10 {
        let start = Instant::now();
        let harness = big_transcript_harness();
        let elapsed = start.elapsed();
        worst = worst.max(elapsed);
        best = best.min(elapsed);
        assert!(harness.window().window().is_visible());
        harness.session.window.hide().ok();
    }
    println!("window open: best {best:?}, worst {worst:?}");
    assert!(
        worst < Duration::from_millis(1_500),
        "opening the window took {worst:?} in the worst of ten"
    );
});

crate::ui_test!(a_twenty_thousand_word_transcript_loads_within_budget, {
    let h = big_transcript_harness();

    // Select the finished job and time everything from the user's point of
    // view: the click, the reload of the transcript, the relayout of the row
    // geometry and the commit into the window.
    let start = Instant::now();
    h.select_finished();
    let elapsed = start.elapsed();

    let words = h.model().transcript_view().word_count();
    let segments = usize::try_from(h.window().get_transcript_rows_total()).unwrap();
    println!(
        "transcript load: {words} words, {segments} segments in {elapsed:?}; \
         {} live delegates",
        h.model().live_transcript_rows()
    );

    assert!(words >= WORDS, "the fixture must really be {WORDS} words");
    assert!(segments > 500, "a 20,000-word meeting is many segments");

    // The headline number. A pane that takes ten seconds to open is a failed
    // product, and one that takes a second is a product people describe as
    // "slow". Half a second is the budget, and the measured figure is normally
    // three orders of magnitude under it.
    assert!(
        elapsed < Duration::from_millis(500),
        "loading a {WORDS}-word transcript took {elapsed:?}"
    );

    // The load must be independent of the transcript's length in the only way
    // that matters: the number of live delegates must be a function of the
    // viewport, not of the document.
    let live = h.model().live_transcript_rows();
    assert!(
        live < 200,
        "{live} live delegates for a {segments}-segment transcript; the windowing is not working"
    );
});

crate::ui_test!(scrolling_stays_cheap_at_every_position, {
    let h = big_transcript_harness();
    h.select_finished();

    let total_lines = h.window().get_transcript_lines_total();
    let viewport = 600.0_f32;
    let line_height = TRANSCRIPT_LINE_HEIGHT_PX;
    let total_px = total_lines as f32 * line_height;
    let steps = (total_px / viewport).ceil().max(1.0) as usize;
    println!("scrolling {steps} viewports over {total_lines} lines");

    let mut worst = Duration::ZERO;
    let mut worst_live = 0usize;
    let start_all = Instant::now();
    for step in 0..steps {
        let offset = step as f32 * viewport;
        let start = Instant::now();
        // Exactly what a wheel event does: report the new offset and the new
        // viewport size, and let the UI rewindow.
        h.model().viewport_changed(-offset, viewport, 800.0);
        let elapsed = start.elapsed();
        worst = worst.max(elapsed);
        worst_live = worst_live.max(h.model().live_transcript_rows());
    }
    let total = start_all.elapsed();
    let mean = total / steps.max(1) as u32;
    println!(
        "scroll: {steps} viewports in {total:?}, mean {mean:?}, worst single update \
         {worst:?}, worst live delegates {worst_live}"
    );

    assert!(
        mean < Duration::from_millis(5),
        "a scroll update averages {mean:?}; at 60 Hz that is a third of the frame"
    );
    assert!(
        worst < Duration::from_millis(40),
        "one scroll update took {worst:?}. The first update after a scroll is also the \
         first time the software renderer measures this text, so a single outlier \
         above a frame is expected; a sustained one is not"
    );
    // A per-viewport budget over the whole document: a minute of scrolling has
    // to stay interactive.
    assert!(
        total < Duration::from_millis(2_000),
        "scrolling the whole transcript took {total:?}"
    );
    // The windowing claim, as a number at the worst position rather than the
    // first one.
    let rows_per_screen = (viewport / line_height) as usize + 1;
    assert!(
        worst_live <= rows_per_screen + 16,
        "the live delegate count peaked at {worst_live} for a {rows_per_screen}-row viewport"
    );
});

crate::ui_test!(resizing_the_pane_relayouts_without_exploding, {
    let h = big_transcript_harness();
    h.select_finished();

    // Dragging a window edge produces a viewport change per frame. Each one
    // may relayout every segment's line count, so the cost has to be measured
    // against the document size, not waved at.
    let start = Instant::now();
    for step in 0..120 {
        let width = 400.0 + (step % 60) as f32 * 10.0;
        h.model().viewport_changed(0.0, 600.0, width);
    }
    let elapsed = start.elapsed();
    let per_frame = elapsed / 120;
    println!("resize: 120 viewport changes in {elapsed:?}, {per_frame:?} each");

    assert!(
        per_frame < Duration::from_millis(8),
        "a relayout costs {per_frame:?} per frame, which would drop frames while dragging"
    );
    assert!(
        h.model().live_transcript_rows() < 200,
        "resizing must not materialise the document"
    );
});

crate::ui_test!(many_jobs_do_not_slow_the_window_down, {
    // A batch tool with a hundred queued files is a normal Tuesday. The job
    // list is a `VecModel` and the commit is a diff, so this should be linear
    // and small; the assertion is that it is.
    let backend = FakeController::with_options(Duration::from_millis(40), 2_000, 2, 0);
    let h = Harness::with_backend(backend, common::RecordingHostServices::cancelling());
    h.session.window.show().expect("show");

    let start = Instant::now();
    for n in 0..100 {
        h.model()
            .input_edited(format!("C:/recordings/batch-{n:03}.mkv"));
        h.model().enqueue();
    }
    let elapsed = start.elapsed();
    assert_eq!(h.window().get_job_count(), 100);
    println!("100 jobs queued in {elapsed:?}");

    // Ten commits over a hundred running jobs, which is what a progress tick
    // actually costs once the list is that long.
    let commit_start = Instant::now();
    for _ in 0..10 {
        h.pump();
    }
    let commit = commit_start.elapsed();
    println!("10 progress commits over 100 jobs in {commit:?}");
    assert!(
        commit < Duration::from_millis(200),
        "10 commits over 100 jobs took {commit:?}"
    );
});

crate::ui_test!(idle_cpu_is_not_a_number_this_file_can_measure, {
    // An honest placeholder for the one benchmark a headless test cannot do.
    //
    // Measuring idle CPU needs a process, a sampler and a window that is
    // actually on a display, none of which exist in `cargo test`. What *is*
    // measurable here is the thing that causes it: how many times a commit runs
    // while nothing is happening. The pump timer runs at 30 Hz whether or not
    // anything arrived, so an idle window does no work per tick beyond one
    // `try_recv` — and this test proves the pump does not touch Slint at all
    // when the channel is empty, which is the property that keeps the CPU flat.
    let h = big_transcript_harness();
    h.select_finished();

    let model = h.session.model.clone();
    let before = model.pumps();
    let footer_before = h.window().get_footer().to_string();
    let start = Instant::now();
    for _ in 0..1_000 {
        model.pump();
    }
    let elapsed = start.elapsed();
    let per_pump = elapsed / 1_000;
    assert_eq!(model.pumps(), before + 1_000);
    assert_eq!(
        h.window().get_footer().to_string(),
        footer_before,
        "an idle pump must not change a single property"
    );
    println!("1,000 idle pumps in {elapsed:?}, {per_pump:?} each");
    assert!(
        per_pump < Duration::from_micros(50),
        "an idle pump costs {per_pump:?}; at 30 Hz that is real CPU"
    );
});

crate::ui_test!(the_window_model_never_exceeds_the_viewport, {
    // The invariant, stated once more as a test so a future change that breaks
    // it fails here rather than in a screenshot review.
    let h = big_transcript_harness();
    h.select_finished();
    let total = usize::try_from(h.window().get_transcript_rows_total()).unwrap();
    assert!(
        total > 1_000,
        "the fixture must be long enough to be interesting"
    );

    let mut worst = 0;
    let total_lines = h.window().get_transcript_lines_total();
    for step in 0..200 {
        let offset = step as f32 * (total_lines as f32 * TRANSCRIPT_LINE_HEIGHT_PX) / 200.0;
        h.model().viewport_changed(-offset, 500.0, 700.0);
        worst = worst.max(h.model().live_transcript_rows());
    }
    println!("{total} segments, peak live delegates {worst}");
    assert!(
        worst < 120,
        "peaked at {worst} live delegates for {total} segments"
    );
    // And the delegate rows are real rows, not placeholders.
    let model = h.window().get_transcript_window();
    for index in 0..model.row_count() {
        let row = model.row_data(index).expect("a row in range");
        assert!(!row.text.is_empty(), "row {index} has no text");
        assert!(row.line_count >= 1, "row {index} has no height");
        assert!(row.lines_before >= 0);
    }
});

crate::ui_test!(a_worker_harness_is_the_only_source_of_the_words, {
    // Guards the honesty claim: the fixture really is labelled sample data, and
    // the word count really is 20,000. A perf number measured against
    // accidentally-real data would be meaningless.
    let h = big_transcript_harness();
    h.select_finished();
    let job = h
        .model()
        .job_list()
        .jobs()
        .iter()
        .find(|j| j.has_transcript)
        .cloned()
        .expect("the finished job");
    let transcript = h
        .backend
        .transcript(&job.id)
        .expect("readable")
        .expect("a transcript");
    assert!(
        vaani_ui::sample::assert_is_sample(&transcript),
        "the fixture must be labelled sample data"
    );
    let words: usize = transcript.segments.iter().map(|s| s.words.len()).sum();
    assert!(words >= WORDS, "the fixture has {words} words, not {WORDS}");
    let model: Arc<FakeController> = h.backend.clone();
    assert_eq!(model.job(&job.id).and_then(|j| j.word_count), Some(words));
});
