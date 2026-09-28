//! Clearly-labelled sample transcripts.
//!
//! # Why this module exists at all
//!
//! The pipeline is not written yet, so the UI needs *something* to render. The
//! temptation is to write plausible-looking meeting text. That is a trap: a
//! screenshot of a mock transcript is indistinguishable from a screenshot of a
//! broken real one, and the person debugging it at 2am will be the person who
//! assumed the mock was real.
//!
//! So every word produced here is drawn from a fixed, obviously-contrived
//! vocabulary, and every segment is prefixed with `[sample segment N]`. There
//! is no way to confuse this for a transcription result, and
//! [`assert_is_sample`] exists so a test can prove a transcript came from here.

use std::sync::Arc;

use vaani_core::{EngineStamp, Segment, Speaker, Time, Transcript, Word};

/// Prefix on every generated segment's text.
pub const SAMPLE_SEGMENT_PREFIX: &str = "[sample segment";

/// Prefix on the whole-transcript marker used in the UI footer.
pub const SAMPLE_MARKER: &str = "[sample data";

/// The vocabulary. Deliberately not words anyone would say in a meeting.
const VOCABULARY: [&str; 24] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray",
];

/// Average words per generated segment. A real ASR segment is a sentence or a
/// clause; this is in the same range so the layout is exercised realistically.
const WORDS_PER_SEGMENT: usize = 11;

/// The `EngineStamp` written into sample transcripts.
///
/// Says `sample` in every field, so an exported sample transcript names its
/// own origin in the file.
pub fn sample_engine() -> EngineStamp {
    EngineStamp {
        asr_model: "sample".to_owned(),
        asr_version: "0".to_owned(),
        diarization_model: Some("sample".to_owned()),
        structuring_model: None,
    }
}

/// Word number `i` of the sample vocabulary, cycling.
///
/// Deterministic so two runs of the same generator produce byte-identical
/// transcripts, which is what makes the performance numbers comparable.
#[must_use]
pub fn sample_word(i: usize) -> &'static str {
    VOCABULARY[i % VOCABULARY.len()]
}

/// Build a sample transcript of roughly `word_count` words.
///
/// The result is deterministic for a given `(word_count, speakers)`. Speaker
/// attribution rotates so every speaker has a non-zero talk time, because a
/// speaker panel where one bar is always zero is not worth testing.
#[must_use]
pub fn sample_transcript(word_count: usize, speakers: usize) -> Transcript {
    let speaker_count = speakers.max(1);
    let words_per_segment = WORDS_PER_SEGMENT.max(1);
    let segment_count = word_count.div_ceil(words_per_segment).max(1);

    let mut transcript = Transcript::new(sample_engine());
    // A fixed id, because a random one would make two runs of the same request
    // differ and every determinism claim about this module would be a lie.
    transcript.id = "sample-transcript".to_owned();

    for s in 0..speaker_count {
        transcript.add_speaker(Speaker::new(s));
    }

    // ~2.6 words per second is ordinary conversational pace; the point is only
    // that the timestamps are plausible and monotonic.
    let per_segment_ms: i64 = (words_per_segment as f64 * 385.0) as i64;

    for index in 0..segment_count {
        let start_ms = index as i64 * per_segment_ms;
        let end_ms = start_ms + per_segment_ms - 40;
        let speaker = index % speaker_count;

        // The label goes in as a word, because `Segment::text` is derived from
        // its words. Writing the label anywhere else would leave the transcript
        // looking like real speech in every place the UI actually reads it,
        // which is the exact failure this module exists to prevent.
        let mut words: Vec<Word> = vec![Word::new(
            format!("{SAMPLE_SEGMENT_PREFIX} {index}]"),
            Time::from_millis(start_ms),
            Time::from_millis(start_ms + 40),
        )];
        words.extend((0..words_per_segment).map(|w| {
            let position = index * words_per_segment + w;
            let word_start = start_ms + (w as i64 * (per_segment_ms / words_per_segment as i64));
            Word::new(
                sample_word(position),
                Time::from_millis(word_start),
                Time::from_millis(word_start + (per_segment_ms / words_per_segment as i64) - 30),
            )
        }));

        let mut segment = Segment::new(Time::from_millis(start_ms), Time::from_millis(end_ms))
            .with_words(words)
            .with_speaker(Some(speaker));
        segment.id = format!("sample-seg-{index:06}");
        transcript.segments.push(segment);
    }

    // The *length*: the sample is a whole recording starting at zero, so its
    // extent and its length are the same figure.
    transcript.duration = Some(vaani_core::MediaLength::from_millis(
        segment_count as i64 * per_segment_ms,
    ));
    transcript.recompute_speaker_shares();
    transcript
}

/// Build a sample transcript of roughly `word_count` words behind an `Arc`,
/// which is how the controller hands one out.
#[must_use]
pub fn sample_transcript_arc(word_count: usize, speakers: usize) -> Arc<Transcript> {
    Arc::new(sample_transcript(word_count, speakers))
}

/// Whether `transcript` came from this module.
///
/// Used by tests to assert that no fake run can be mistaken for a real one,
/// and by the fake backend's own self-check.
#[must_use]
pub fn assert_is_sample(transcript: &Transcript) -> bool {
    transcript.engine.asr_model == "sample"
        || transcript
            .segments
            .first()
            .is_some_and(|s| s.text().starts_with(SAMPLE_SEGMENT_PREFIX))
}

/// The `SpeakerSummary` list matching a sample transcript.
#[must_use]
pub fn sample_speakers(transcript: &Transcript) -> Vec<crate::backend::SpeakerSummary> {
    let mut totals = vec![0i64; transcript.speakers.len()];
    for segment in &transcript.segments {
        if let Some(speaker) = segment.speaker
            && let Some(slot) = totals.get_mut(speaker)
        {
            *slot += segment.duration().as_millis().max(0);
        }
    }
    let total: i64 = totals.iter().sum();
    transcript
        .speakers
        .iter()
        .map(|speaker| crate::backend::SpeakerSummary {
            index: speaker.index,
            name: speaker.name.clone(),
            talk_time_ms: totals.get(speaker.index).copied().unwrap_or(0),
            share: speaker.share.or_else(|| {
                let spoken = totals.get(speaker.index).copied().unwrap_or(0);
                (total > 0).then_some(spoken as f32 / total as f32)
            }),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let a = sample_transcript(500, 3);
        let b = sample_transcript(500, 3);
        assert_eq!(a, b, "the same request must produce the same transcript");
    }

    #[test]
    fn every_segment_is_labelled() {
        let t = sample_transcript(200, 2);
        for (i, segment) in t.segments.iter().enumerate() {
            let text = segment.text();
            assert!(
                text.starts_with(SAMPLE_SEGMENT_PREFIX),
                "segment {i} is not labelled: {text:?}"
            );
            assert!(
                text.contains(&format!("{i}]")),
                "segment {i} has the wrong label"
            );
        }
    }

    #[test]
    fn assert_is_sample_recognises_our_own_output() {
        assert!(assert_is_sample(&sample_transcript(50, 2)));
        let real = Transcript::new(EngineStamp::asr("sravaani", "1.0"));
        assert!(!assert_is_sample(&real));
    }

    #[test]
    fn word_count_is_close_to_what_was_asked_for() {
        let t = sample_transcript(20_000, 4);
        // One extra word per segment is the label, so the total is the request
        // plus the segment count. A test that forgot the label would pass this
        // and then fail every label assertion, so both are asserted.
        let segments = t.segments.len();
        let words: usize = t.segments.iter().map(|s| s.words.len()).sum();
        // Eleven spoken words plus one label word per segment.
        assert_eq!(
            words,
            segments * (WORDS_PER_SEGMENT + 1),
            "every segment carries its label word, and every segment has the same shape"
        );
        assert!(
            words >= 20_000,
            "{words} words is not a 20,000-word transcript"
        );
    }

    #[test]
    fn timestamps_are_monotonic_and_within_the_duration() {
        let t = sample_transcript(5_000, 3);
        let mut last = i64::MIN;
        for segment in &t.segments {
            assert!(segment.start.as_millis() >= last, "segments out of order");
            assert!(segment.end >= segment.start, "inverted segment");
            last = segment.start.as_millis();
        }
        // `duration` is a **length** and a segment's `end` is a **position**.
        // They were compared directly here before, which is only correct for a
        // sample that starts at zero. The sample does, so the assertion is
        // unchanged in meaning — but it is now stated through the window, which
        // is the expression that is correct in general.
        let window =
            vaani_core::MediaWindow::of_length(t.duration.expect("the sample declares one"));
        assert_eq!(window.start(), Time::ZERO);
        assert!(t.segments.last().unwrap().end <= window.end());
    }

    #[test]
    fn every_speaker_gets_a_non_zero_talk_time() {
        let t = sample_transcript(1_000, 4);
        let speakers = sample_speakers(&t);
        assert_eq!(speakers.len(), 4);
        for s in &speakers {
            assert!(s.talk_time_ms > 0, "speaker {} never spoke", s.index);
        }
        let sum: f32 = speakers.iter().filter_map(|s| s.share).sum();
        assert!((sum - 1.0).abs() < 1e-4, "shares summed to {sum}");
    }

    #[test]
    fn a_single_speaker_transcript_is_still_well_formed() {
        let t = sample_transcript(100, 0);
        assert_eq!(t.speakers.len(), 1);
        assert!(t.segments.iter().all(|s| s.speaker == Some(0)));
    }
}
