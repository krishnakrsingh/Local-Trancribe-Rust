//! Aggregate statistics over a transcript, for the report and the settings panel.
//!
//! Grouping is a numbers decision, so the numbers have to be readable without
//! a debugger. [`stats`] gives the distribution a product owner actually asks
//! about — how long a segment lasts, how many words it holds — as plain
//! aggregates with no transcript content in it, which is what makes it safe to
//! print on a real recording.

use vaani_core::transcript::Transcript;

/// Counts and duration distribution of a transcript's segments.
///
/// All fields are aggregates. Nothing here holds, returns or can print a word.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TranscriptStats {
    /// Segments in the transcript.
    pub segments: usize,
    /// Words in the transcript.
    pub words: usize,
    /// Segments with no words. Always 0 in a grouped transcript.
    pub empty_segments: usize,
    /// Segments with a speaker.
    pub attributed_segments: usize,
    /// Distinct speakers owning at least one segment.
    pub speakers_used: usize,
    /// Mean words per segment.
    pub mean_words_per_segment: f64,
    /// Mean characters of segment text, words joined by single spaces.
    pub mean_chars_per_segment: f64,
    /// Mean segment duration, in milliseconds.
    pub mean_duration_ms: f64,
    /// Median segment duration, in milliseconds.
    pub median_duration_ms: i64,
    /// 90th-percentile segment duration, in milliseconds.
    pub p90_duration_ms: i64,
    /// Longest segment duration, in milliseconds.
    pub max_duration_ms: i64,
    /// Shortest segment duration, in milliseconds.
    pub min_duration_ms: i64,
    /// Segments longer than the caption convention's comfortable maximum. The
    /// threshold is 7 s, which is where a two-line caption stops being readable
    /// at normal speaking rate.
    pub segments_over_7s: usize,
}

impl TranscriptStats {
    /// Words per segment, or 0 for an empty transcript.
    #[must_use]
    pub fn words_per_segment(&self) -> f64 {
        if self.segments == 0 {
            0.0
        } else {
            self.words as f64 / self.segments as f64
        }
    }
}

/// Nearest-rank percentile of an ascending slice: the smallest value at or
/// above which `p` of the data lies.
///
/// Nearest-rank rather than an interpolated percentile because every element
/// here is an integer millisecond count and an interpolated 90th percentile of
/// a duration distribution is a number that corresponds to no segment.
#[must_use]
pub fn percentile(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    let p = p.clamp(0.0, 1.0);
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Summarise `transcript`. Pure, and never panics on an empty one.
#[must_use]
pub fn stats(t: &Transcript) -> TranscriptStats {
    let segments = t.segments.len();
    let words: usize = t.segments.iter().map(|s| s.words.len()).sum();
    let empty_segments = t.segments.iter().filter(|s| s.words.is_empty()).count();
    let attributed_segments = t.segments.iter().filter(|s| s.speaker.is_some()).count();

    let mut speakers: Vec<usize> = t.segments.iter().filter_map(|s| s.speaker).collect();
    speakers.sort_unstable();
    speakers.dedup();

    if segments == 0 {
        return TranscriptStats {
            segments,
            words,
            empty_segments,
            attributed_segments,
            speakers_used: speakers.len(),
            ..TranscriptStats::default()
        };
    }

    let mut durations: Vec<i64> = t
        .segments
        .iter()
        .map(|s| s.duration().as_millis().max(0))
        .collect();
    let total_duration: i64 = durations.iter().sum();
    let total_chars: usize = t
        .segments
        .iter()
        .map(|s| {
            s.words
                .iter()
                .map(|w| w.display_text().chars().count())
                .sum::<usize>()
                + s.words.len().saturating_sub(1)
        })
        .sum();

    durations.sort_unstable();
    let max_duration_ms = *durations.last().unwrap_or(&0);
    let min_duration_ms = *durations.first().unwrap_or(&0);

    TranscriptStats {
        segments,
        words,
        empty_segments,
        attributed_segments,
        speakers_used: speakers.len(),
        mean_words_per_segment: words as f64 / segments as f64,
        mean_chars_per_segment: total_chars as f64 / segments as f64,
        mean_duration_ms: total_duration as f64 / segments as f64,
        median_duration_ms: percentile(&durations, 0.50),
        p90_duration_ms: percentile(&durations, 0.90),
        max_duration_ms,
        min_duration_ms,
        segments_over_7s: durations.iter().filter(|d| **d > 7_000).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vaani_core::transcript::{EngineStamp, Segment, Time, Word};

    fn t(ms: i64) -> Time {
        Time::from_millis(ms)
    }

    fn transcript_of(spans: &[(i64, i64, usize)]) -> Transcript {
        let mut tr = Transcript::new(EngineStamp::asr("test", "1"));
        for (i, (start, end, n)) in spans.iter().enumerate() {
            let words: Vec<Word> = (0..*n)
                .map(|_| Word::new("ab", t(*start), t(*end)))
                .collect();
            let mut seg = Segment::new(t(*start), t(*end)).with_words(words);
            seg.id = format!("s{i}");
            tr.segments.push(seg);
        }
        tr
    }

    #[test]
    fn an_empty_transcript_is_all_zeroes_and_does_not_panic() {
        let s = stats(&Transcript::new(EngineStamp::asr("t", "1")));
        assert_eq!(s.segments, 0);
        assert_eq!(s.words, 0);
        assert_eq!(s.mean_duration_ms, 0.0);
        assert_eq!(s.median_duration_ms, 0);
        assert_eq!(s.p90_duration_ms, 0);
        assert_eq!(s.words_per_segment(), 0.0);
    }

    #[test]
    fn the_distribution_is_over_segments_not_words() {
        let tr = transcript_of(&[(0, 1_000, 1), (1_000, 3_000, 5), (3_000, 4_000, 1)]);
        let s = stats(&tr);
        assert_eq!(s.segments, 3);
        assert_eq!(s.words, 7);
        assert!((s.words_per_segment() - 7.0 / 3.0).abs() < 1e-9);
        // Durations 1000, 2000, 1000.
        assert!((s.mean_duration_ms - 4000.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.median_duration_ms, 1_000);
        assert_eq!(s.p90_duration_ms, 2_000);
        assert_eq!(s.min_duration_ms, 1_000);
        assert_eq!(s.max_duration_ms, 2_000);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let v: Vec<i64> = (1..=10).collect();
        assert_eq!(percentile(&v, 0.0), 1);
        assert_eq!(percentile(&v, 0.5), 5);
        assert_eq!(percentile(&v, 0.9), 9);
        assert_eq!(percentile(&v, 1.0), 10);
        assert_eq!(percentile(&[], 0.9), 0);
        // Out-of-range p is clamped rather than panicking.
        assert_eq!(percentile(&v, 2.0), 10);
        assert_eq!(percentile(&v, -1.0), 1);
    }

    #[test]
    fn empty_segments_and_attribution_are_counted_separately() {
        let mut tr = transcript_of(&[(0, 1_000, 2), (1_000, 2_000, 0)]);
        tr.segments[0].speaker = Some(0);
        tr.segments[1].speaker = Some(1);
        tr.segments[1].speaker = None;
        let s = stats(&tr);
        assert_eq!(s.empty_segments, 1);
        assert_eq!(s.attributed_segments, 1);
        assert_eq!(s.speakers_used, 1);
    }

    #[test]
    fn segments_longer_than_seven_seconds_are_counted() {
        let tr = transcript_of(&[(0, 8_000, 1), (8_000, 9_000, 1)]);
        assert_eq!(stats(&tr).segments_over_7s, 1);
    }
}
