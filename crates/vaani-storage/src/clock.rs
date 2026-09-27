//! Wall-clock helpers.
//!
//! Times are stored as **integer milliseconds since the Unix epoch**, never as
//! text and never as a SQLite `datetime` string. Two reasons:
//!
//! * `datetime('now')` is a string, and comparing or sorting it is
//!   lexicographic. It happens to be right for ISO-8601 and it is wrong for
//!   anything else, including every hand-written migration.
//! * Milliseconds are what [`vaani_core::Time`] already uses, so a wall-clock
//!   stamp and a media offset are the same type of number and cannot be
//!   confused by a caller that forgets which one it has.

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// Milliseconds since the Unix epoch, UTC.
pub fn now_ms() -> i64 {
    let now = OffsetDateTime::now_utc();
    // `unix_timestamp() * 1000 + ms` rather than `unix_timestamp_nanos()`,
    // which panics outside 1678..2262 and a transcript timestamp does not
    // care that much.
    now.unix_timestamp() * 1_000 + i64::from(now.millisecond())
}

/// Format epoch milliseconds as RFC 3339, or `None` if the value is not a
/// representable instant. Used for display and for the export log; never for
/// comparison.
pub fn rfc3339(ms: i64) -> Option<String> {
    OffsetDateTime::from_unix_timestamp_nanos(i128::from(ms) * 1_000_000)
        .ok()
        .and_then(|t| t.format(&Rfc3339).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_is_this_century_and_in_milliseconds() {
        let now = now_ms();
        // 2020-01-01T00:00:00Z .. 2100-01-01T00:00:00Z
        assert!(
            (1_577_836_800_000..4_102_444_800_000).contains(&now),
            "now_ms() = {now} is not a plausible current time"
        );
    }

    #[test]
    fn rfc3339_round_trips_a_known_instant() {
        assert_eq!(
            rfc3339(0).as_deref(),
            Some("1970-01-01T00:00:00Z"),
            "the epoch must format as the epoch"
        );
        assert_eq!(
            rfc3339(1_700_000_000_123).as_deref(),
            Some("2023-11-14T22:13:20.123Z")
        );
    }

    #[test]
    fn an_absurd_timestamp_is_none_not_a_panic() {
        assert_eq!(rfc3339(i64::MAX), None);
        assert_eq!(rfc3339(i64::MIN / 2), None);
    }
}
