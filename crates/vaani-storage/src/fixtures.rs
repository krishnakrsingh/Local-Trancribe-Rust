//! Deterministic generators for tests, shared by this crate's tests and by
//! `vaani-export`'s.
//!
//! # Why this is in the library
//!
//! Two crates need the same awkward inputs — a transcript with a segment too
//! long for a subtitle cue, a word whose end precedes its start, a mix of
//! Devanagari and Latin and CJK, a speaker with a rename and one without. They
//! are in different crates, and a `dev-dependency` on a shared test crate would
//! mean a manifest change the architect owns. So the generator lives here, is
//! `#[doc(hidden)]`, has no dependencies, and both crates use it. A10's QA
//! suite can use it too, which is the point.
//!
//! # Determinism
//!
//! Everything is driven by a seeded SplitMix64, and every "random" choice is
//! made in a fixed order. A failing case is therefore reproducible from its seed
//! alone, which is the only property that makes a property test worth having:
//! `transcript(0xC0FFEE, 250)` fails the same way tomorrow.

use vaani_core::{
    EngineStamp, MediaLength, Segment, Speaker, Time, Transcript, TranscriptMode, Word,
};

/// A seeded SplitMix64. Small, fast, and — unlike `rand` — already in the
/// dependency budget.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator for `seed`.
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which SplitMix64 handles but which makes
        // "seed 0" indistinguishable from "unseeded" in a failure message.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, n)`. `n == 0` yields 0.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }

    /// A value in `[lo, hi]`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        lo + self.below((hi - lo + 1) as u64) as i64
    }

    /// `true` with probability `numerator / denominator`.
    pub fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.below(denominator.max(1)) < numerator
    }

    /// One element of `items`.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }

    /// An `f32` in `[0, 1)`, or `None` one time in four — engines report
    /// confidence for some words and not others, and the store must preserve
    /// the difference between "0.0" and "absent".
    pub fn confidence(&mut self) -> Option<f32> {
        if self.chance(1, 4) {
            None
        } else {
            Some(self.next_u64() as f32 / u64::MAX as f32)
        }
    }
}

/// Word material, chosen to break exporters on purpose.
///
/// Every entry here is something a real meeting transcript contains and every
/// naive formatter mangles: Devanagari, CJK, an emoji with a zero-width
/// joiner, characters that are markup in Markdown, characters that are markup
/// in SRT, a JSON escape, a 300-character token that must not be split.
const WORDS: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "transcription",
    "deadline",
    "okay",
    "hmm",
    "2026-09-27",
    "C:\\Users\\krish",
    "don't",
    "naïve",
    "café",
    "नमस्ते",
    "भारत",
    "你好",
    "こんにちは",
    "안녕하세요",
    "#hashtag",
    "*emphasis*",
    "_underscore_",
    "`code`",
    "[link](http://x)",
    "a|b",
    "> quote",
    "1. 100% & 50%",
    "-->",
    "<i>html</i>",
    "\"quoted\"",
    "{}",
    "🎧",
    "👩‍💻",
    "e\u{301}",
    "ß",
    "½",
    "supercalifragilisticexpialidocious-extraordinarily-long-hyphenated-token-that-no-wrapper-can-break",
    "(parenthetical)",
    "[bracket]",
    "{brace}",
    "|pipe|",
    "\\backslash\\",
    "\ttab",
    "a  double  space",
];

/// A transcript with `segments` segments, deterministic in `seed`.
///
/// It deliberately contains the shapes that break code:
///
/// * empty segments (no words) and single-word segments;
/// * a segment whose words run past the next segment's start (overlap), which
///   the subtitle exporters must serialise rather than emit;
/// * a word whose end precedes its start, which the store must preserve rather
///   than silently clamp;
/// * negative start times, which `Time` is signed for;
/// * segments with and without a speaker, and speakers with and without a name;
/// * mixed scripts, markup characters and a token longer than any line width.
pub fn transcript(seed: u64, segments: usize) -> Transcript {
    let mut rng = Rng::new(seed);
    let mut t = Transcript::new(EngineStamp {
        asr_model: format!("sravaani-{}", rng.below(3)),
        asr_version: format!("1.{}.{}", rng.below(4), rng.below(10)),
        diarization_model: rng.chance(1, 2).then(|| "pyannote-3.1".to_string()),
        structuring_model: rng.chance(1, 4).then(|| "qwen2.5-7b".to_string()),
    });
    t.id = format!("tr-{seed:016x}");
    t.duration = rng
        .chance(4, 5)
        .then(|| MediaLength::from_millis(rng.range(0, 3_600_000)));
    t.mode = match rng.below(3) {
        0 => TranscriptMode::Raw,
        1 => TranscriptMode::Clean,
        _ => TranscriptMode::Structured,
    };
    t.edited = rng.chance(1, 5);

    let speaker_count = 1 + rng.below(4) as usize;
    // Names are *custom* names only. `Speaker::display_name()` already
    // synthesises "Speaker N" when `name` is `None`, and a transcript that
    // stored the generated label in the name field could never fall back to it
    // — so the generator uses plausible participant names and leaves the rest
    // unnamed, which is the real distribution.
    const NAMES: &[&str] = &["Krishna", "Meera", "Ananya", "Rahul", "Fatima"];
    let mut speakers: Vec<Speaker> = (0..speaker_count)
        .map(|i| Speaker {
            index: i,
            name: rng.chance(1, 2).then(|| (*rng.pick(NAMES)).to_string()),
            share: rng.confidence(),
        })
        .collect();
    // Shuffled on purpose: `speakers` is a `Vec`, and the store must give back
    // the same order it was given rather than sorting by index.
    for i in (1..speakers.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        speakers.swap(i, j);
    }
    t.speakers = speakers;

    let mut cursor = rng.range(0, 30_000);
    for i in 0..segments {
        let span = rng.range(150, 5_000);
        let mut start = cursor;
        // Every so often a segment starts before the previous one ended. ASR
        // does this across chunk boundaries, and both the store and the
        // exporters have to cope.
        if i > 0 && rng.chance(1, 8) {
            start = (cursor - rng.range(1, 400)).max(0);
        }
        if rng.chance(1, 30) {
            start = -rng.range(1, 50);
        }
        let mut end = start + span;
        let n_words = match rng.below(10) {
            0 => 0,
            1 | 2 => 1,
            n => n as usize,
        };
        let mut words = Vec::with_capacity(n_words);
        let mut w = start;
        for _ in 0..n_words {
            let text = *rng.pick(WORDS);
            let wlen = rng.range(80, 900);
            let mut word_end = w + wlen;
            if rng.chance(1, 40) {
                // Degenerate: the engine reported a word that ends before it
                // starts. `push_chunk` fixes these in the pipeline, but a value
                // that reaches the store must survive a round trip untouched.
                word_end = w - rng.range(1, 100);
            }
            if rng.chance(1, 20) {
                word_end = w;
            }
            let normalized = if rng.chance(1, 12) {
                Some(text.to_uppercase())
            } else {
                None
            };
            words.push(Word {
                text: text.to_string(),
                start: Time(w),
                end: Time(word_end),
                confidence: rng.confidence(),
                normalized,
            });
            w = word_end.max(w) + rng.range(0, 60);
        }
        if !words.is_empty() {
            end = end.max(w);
        }
        cursor = end + rng.range(0, 900);
        t.segments.push(Segment {
            id: format!("seg-{seed:016x}-{i:06}"),
            start: Time(start),
            end: Time(end),
            speaker: if rng.chance(3, 10) {
                None
            } else {
                Some(rng.below(speaker_count as u64) as usize)
            },
            language: match rng.below(8) {
                0..=3 => None,
                4 => Some("hi".into()),
                5 => Some("en".into()),
                6 => Some("hinglish".into()),
                _ => Some("other".into()),
            },
            words,
            confidence: rng.confidence(),
        });
    }
    t
}

/// Any word in the transcript, for building a search that must match.
pub fn any_word(t: &Transcript) -> String {
    t.segments
        .iter()
        .flat_map(|s| s.words.iter())
        .map(|w| w.text.clone())
        .next()
        .unwrap_or_else(|| "nothing".to_string())
}

/// A transcript with no segments at all: the empty-file case.
pub fn empty() -> Transcript {
    Transcript::new(EngineStamp::asr("sravaani", "1.0"))
}

/// The first difference between two transcripts, as one readable line.
///
/// A failed `assert_eq!` on a 1,800-segment transcript prints both sides in full
/// — megabytes of `Debug` output in which the actual difference is invisible.
/// This walks the structures in a fixed order and stops at the first mismatch,
/// naming it precisely: the segment index, the field, and both values.
///
/// Returns `None` when the two are equal.
pub fn diff(expected: &Transcript, actual: &Transcript) -> Option<String> {
    if expected.id != actual.id {
        return Some(format!("id: {:?} != {:?}", expected.id, actual.id));
    }
    if expected.duration != actual.duration {
        return Some(format!(
            "duration: {:?} != {:?}",
            expected.duration, actual.duration
        ));
    }
    if expected.mode != actual.mode {
        return Some(format!("mode: {:?} != {:?}", expected.mode, actual.mode));
    }
    if expected.edited != actual.edited {
        return Some(format!(
            "edited: {:?} != {:?}",
            expected.edited, actual.edited
        ));
    }
    if expected.engine != actual.engine {
        return Some(format!(
            "engine: {:?} != {:?}",
            expected.engine, actual.engine
        ));
    }
    if expected.segments.len() != actual.segments.len() {
        return Some(format!(
            "segment count: {} != {}",
            expected.segments.len(),
            actual.segments.len()
        ));
    }
    for (i, (e, a)) in expected
        .segments
        .iter()
        .zip(actual.segments.iter())
        .enumerate()
    {
        if e.id != a.id {
            return Some(format!("segment[{i}].id: {:?} != {:?}", e.id, a.id));
        }
        let segment_id = &e.id;
        if e.start != a.start {
            return Some(format!(
                "segment[{i}] {:?}.start: {} != {}",
                e.id,
                e.start.as_millis(),
                a.start.as_millis()
            ));
        }
        if e.end != a.end {
            return Some(format!(
                "segment[{i}] {:?}..end: {} != {}",
                e.id,
                e.end.as_millis(),
                a.end.as_millis()
            ));
        }
        if e.speaker != a.speaker {
            return Some(format!(
                "segment[{i}] {:?}.speaker: {:?} != {:?}",
                e.id, e.speaker, a.speaker
            ));
        }
        if e.language != a.language {
            return Some(format!(
                "segment[{i}] {:?}.language: {:?} != {:?}",
                e.id, e.language, a.language
            ));
        }
        if e.confidence != a.confidence {
            return Some(format!(
                "segment[{i}] {:?}.confidence: {:?} != {:?}",
                e.id, e.confidence, a.confidence
            ));
        }
        if e.text() != a.text() {
            return Some(format!(
                "segment[{i}] {:?} text: {:?} != {:?}",
                e.id,
                e.text(),
                a.text()
            ));
        }
        if e.words.len() != a.words.len() {
            return Some(format!(
                "segment[{i}] {:?} word count: {} != {}",
                e.id,
                e.words.len(),
                a.words.len()
            ));
        }
        for (w, (e, a)) in e.words.iter().zip(a.words.iter()).enumerate() {
            let at = format!("segment[{} {:?}].words[{w}]", i, segment_id);
            if e.text != a.text {
                return Some(format!("{at}.text: {:?} != {:?}", e.text, a.text));
            }
            if e.normalized != a.normalized {
                return Some(format!(
                    "{at}.normalized: {:?} != {:?}",
                    e.normalized, a.normalized
                ));
            }
            if e.start != a.start {
                return Some(format!(
                    "{at}.start: {} != {}",
                    e.start.as_millis(),
                    a.start.as_millis()
                ));
            }
            if e.end != a.end {
                return Some(format!(
                    "{at}.end: {} != {}",
                    e.end.as_millis(),
                    a.end.as_millis()
                ));
            }
            if e.confidence != a.confidence {
                return Some(format!(
                    "{at}.confidence: {:?} != {:?}",
                    e.confidence, a.confidence
                ));
            }
        }
    }
    if expected.speakers.len() != actual.speakers.len() {
        return Some(format!(
            "speaker count: {} != {}",
            expected.speakers.len(),
            actual.speakers.len()
        ));
    }
    for (i, (e, a)) in expected
        .speakers
        .iter()
        .zip(actual.speakers.iter())
        .enumerate()
    {
        if e.index != a.index {
            return Some(format!("speakers[{i}].index: {} != {}", e.index, a.index));
        }
        if e.name != a.name {
            return Some(format!(
                "speakers[{i}] {:?}.name: {:?} != {:?}",
                e.index, e.name, a.name
            ));
        }
        if e.share != a.share {
            return Some(format!(
                "speakers[{i}] {:?} share: {:?} != {:?}",
                e.index, e.share, a.share
            ));
        }
    }
    None
}

/// Assert two transcripts are equal, reporting the first difference.
///
/// # Panics
///
/// With the difference described in one line, or with "identical" if they are.
pub fn assert_transcript_eq(expected: &Transcript, actual: &Transcript) {
    if let Some(d) = diff(expected, actual) {
        panic!("transcripts differ: {d}");
    }
}

/// A transcript with exactly one word.
pub fn single_word() -> Transcript {
    let mut t = empty();
    t.segments
        .push(Segment::new(Time(0), Time(500)).with_words(vec![Word::new(
            "हाँ",
            Time(0),
            Time(500),
        )]));
    t
}

/// A transcript whose only segment is far too long for one subtitle cue.
pub fn one_long_segment(words: usize) -> Transcript {
    let mut t = empty();
    let mut w = Time(0);
    let mut list = Vec::with_capacity(words);
    for i in 0..words {
        let start = w;
        let end = Time(start.as_millis() + 300);
        list.push(Word::new(format!("word{i}"), start, end));
        w = end;
    }
    t.segments.push(Segment::new(Time(0), w).with_words(list));
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_transcript() {
        assert_eq!(transcript(7, 20), transcript(7, 20));
        assert_ne!(transcript(7, 20), transcript(8, 20));
    }

    #[test]
    fn the_generator_produces_the_awkward_shapes_it_claims_to() {
        let mut saw_empty_segment = false;
        let mut saw_no_speaker = false;
        let mut saw_overlap = false;
        let mut saw_inverted_word = false;
        let mut saw_negative_start = false;
        let mut saw_no_duration = false;
        for seed in 0..40u64 {
            let t = transcript(seed, 60);
            if t.duration.is_none() {
                saw_no_duration = true;
            }
            for (i, s) in t.segments.iter().enumerate() {
                if s.words.is_empty() {
                    saw_empty_segment = true;
                }
                if s.speaker.is_none() {
                    saw_no_speaker = true;
                }
                if s.start.as_millis() < 0 {
                    saw_negative_start = true;
                }
                if s.words.iter().any(|w| w.end < w.start) {
                    saw_inverted_word = true;
                }
                if let Some(next) = t.segments.get(i + 1)
                    && next.start < s.end
                {
                    saw_overlap = true;
                }
            }
        }
        for (what, seen) in [
            ("an empty segment", saw_empty_segment),
            ("a segment with no speaker", saw_no_speaker),
            ("overlapping segments", saw_overlap),
            ("a word that ends before it starts", saw_inverted_word),
            ("a negative start time", saw_negative_start),
            ("a transcript with no duration", saw_no_duration),
        ] {
            assert!(seen, "the generator never produced {what}");
        }
    }

    #[test]
    fn the_generator_produces_every_shape_of_awkward_text() {
        let mut all = String::new();
        for seed in 0..30u64 {
            for s in transcript(seed, 40).segments {
                for w in s.words {
                    all.push_str(&w.text);
                    all.push(' ');
                }
            }
        }
        for needed in [
            "नमस्ते",
            "你好",
            "-->",
            "<i>",
            "C:\\",
            "🎧",
            "don't",
            "e\u{301}",
        ] {
            assert!(all.contains(needed), "no sample contained {needed:?}");
        }
    }

    #[test]
    fn segment_ids_are_unique_within_a_transcript() {
        let t = transcript(3, 500);
        let mut ids: Vec<&str> = t.segments.iter().map(|s| s.id.as_str()).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before);
    }

    #[test]
    fn rng_is_uniform_enough_to_exercise_every_branch() {
        let mut rng = Rng::new(1);
        let mut seen_low = 0;
        let mut seen_high = 0;
        for _ in 0..200 {
            let v = rng.below(10);
            if v < 5 {
                seen_low += 1;
            } else {
                seen_high += 1;
            }
        }
        assert!(
            seen_low > 70 && seen_high > 70,
            "split was {seen_low}/{seen_high}"
        );
    }

    #[test]
    fn rng_bounds_are_inclusive_and_guarded() {
        let mut rng = Rng::new(2);
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.range(5, 5), 5);
        for _ in 0..100 {
            let v = rng.range(-10, 10);
            assert!((-10..=10).contains(&v));
        }
    }
}
