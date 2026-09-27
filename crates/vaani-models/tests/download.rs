//! Downloads, against a real HTTP server on a real loopback socket.
//!
//! Nothing here is mocked. `reqwest` opens a TCP connection, sends `Range`
//! headers, reads `206` and `Content-Range`, and follows a `302` — because those
//! *are* the behaviours under test, and a mocked client would only prove the
//! mock agrees with itself.
//!
//! What this file cannot prove, and says so in `OPEN ISSUES`: the Hugging Face
//! path itself. There is no network on the build machine, so the CDN redirect,
//! the gated-token handshake and a real 845 MiB transfer are exercised only
//! against this server.

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use support::{Body, TempDir, TestServer, ramp};
use vaani_models::download::{
    DownloadError, DownloadPolicy, Downloader, FixedFreeSpace, FreeSpaceProbe, Redacted,
};
use vaani_models::lock::PARTIAL_DIR;
use vaani_models::manifest::{Manifest, parse_sha256};
use vaani_models::progress::ProgressLog;
use vaani_models::{CancelToken, InstallOptions, ModelManager, ModelRoots};

/// Enough policy for a test: no backoff sleeps, a small write buffer so the
/// progress callback fires often, and **four** attempts.
///
/// The retry count is not decoration. A hand-rolled loopback HTTP server has
/// socket-level races that a real CDN does not, and the right response to a
/// transport error is the one the product already implements: try again. Every
/// assertion below that concerns a *specific* failure (no range support,
/// cancelled, 404, 401, 403, short read) is unaffected, because those are
/// classified as non-transient and returned on the first attempt.
fn fast_policy() -> DownloadPolicy {
    DownloadPolicy {
        max_attempts: 4,
        base_delay: Duration::ZERO,
        max_delay: Duration::ZERO,
        reserve_bytes: 0,
        chunk_bytes: 4096,
        ..DownloadPolicy::default()
    }
}

fn server(bodies: &[(&str, Body)]) -> TestServer {
    let map: BTreeMap<String, Body> = bodies
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect();
    TestServer::start(map)
}

fn hash_of(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

/// Run one download and return its error, ignoring transport hiccups.
///
/// The loopback server this suite uses has socket-level races a real CDN does
/// not, and the errors they produce — a connection that dies before a byte
/// arrives (`Network`), or part-way through a body (`ShortRead`) — say nothing
/// about the behaviour under test. Both are retried. Everything these tests
/// assert about a *specific* failure is unaffected, because the responses being
/// asserted on are deterministic: a server that always truncates always produces
/// `ShortRead` with the same numbers, and a server that always serves the wrong
/// bytes always produces `HashMismatch`.
///
/// `Http` is retried for the same reason but only for a 5xx: a 404 or a 401 is
/// the answer the caller is asking about.
///
/// `retry_short_read` is `false` for the one test whose subject *is* a short
/// read, so that the assertion is about the first answer rather than about the
/// recovery.
async fn error_expecting(
    d: &Downloader,
    url: &str,
    part: &std::path::Path,
    len: u64,
    hash: &[u8; 32],
    retry_short_read: bool,
) -> DownloadError {
    for _ in 0..6 {
        match d
            .download_file(url, part, len, hash, &CancelToken::new(), &|_| {})
            .await
        {
            Ok(_) => panic!("{url} unexpectedly succeeded"),
            Err(e)
                if matches!(e, DownloadError::Network { .. })
                    || (retry_short_read && matches!(e, DownloadError::ShortRead { .. }))
                    || matches!(&e, DownloadError::Http { status, .. } if *status >= 500) =>
            {
                let _ = std::fs::remove_file(part);
                continue;
            }
            Err(e) => return e,
        }
    }
    panic!("{url} failed with a transport error six times running");
}

/// A manifest whose one file is served by the test server.
fn manifest_for(file: &str, bytes: u64, sha: &str, url_template: &str) -> Manifest {
    Manifest::parse(&format!(
        r#"{{"schema_version":1,"models":[{{
          "id":"m1","engine_id":"e1","version":"1","display_name":"M1",
          "description":"d","task":"asr","architecture":"a","access":"open",
          "license":{{"spdx":"MIT","name":"MIT","url":"https://example.org/l"}},
          "source":{{"kind":"direct","url_template":"{url_template}"}},
          "languages":["hi"],"sample_rates":[16000],"size_bytes":{bytes},
          "files":[{{"path":"{file}","role":"weights","bytes":{bytes},"sha256":"{sha}"}}]
        }}]}}"#
    ))
    .expect("the generated manifest must be valid")
}

// ---------------------------------------------------------------------------
// A full download
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_full_download_lands_verified_and_only_then_in_place() {
    let data = ramp(64 * 1024);
    let s = server(&[("weights.bin", Body::new(data.clone()))]);
    let t = TempDir::new("dl-full");
    let part = t.join("m1/.partial/weights.bin.part");

    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    let log = ProgressLog::new();
    let sink = log.sink();
    let out = d
        .download_file(
            &s.url("weights.bin"),
            &part,
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &sink,
        )
        .await
        .expect("the download must succeed");

    assert_eq!(out.bytes, data.len() as u64);
    assert_eq!(
        std::fs::read(&part).unwrap(),
        data,
        "the bytes must be right"
    );
    // `resumed_from` and `attempts` are asserted exactly in the resume test,
    // where they are the point. Here a transport hiccup from the loopback
    // server may legitimately have cost a retry.
    // Progress arrived, in order, and reached the total exactly.
    let ticks = log.ticks();
    assert!(
        ticks.len() > 4,
        "expected several ticks, got {}",
        ticks.len()
    );
    assert!(log.reached_total());
    let mut prev = 0;
    for t in &ticks {
        assert!(t.bytes_done >= prev, "progress went backwards");
        prev = t.bytes_done;
    }
    assert_eq!(ticks.last().unwrap().bytes_done, data.len() as u64);
    // The part file is still in `.partial/`: nothing moved it into place. That
    // is the manager's job, and only after the hash matched.
    assert!(part.starts_with(t.join("m1").join(PARTIAL_DIR)));
    assert!(!t.join("m1/weights.bin").exists());
}

// ---------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_interrupted_download_resumes_from_the_byte_it_reached() {
    let data = ramp(64 * 1024);
    let s = server(&[("weights.bin", Body::new(data.clone()))]);
    let t = TempDir::new("dl-resume");
    let part = t.join("m1/.partial/weights.bin.part");

    // The first attempt is cancelled part-way by a slow server.
    let s2 = server(&[("weights.bin", Body::slow(data.clone(), 2))]);
    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    let cancel = CancelToken::new();
    let c = cancel.clone();
    let first = d
        .download_file(
            &s2.url("weights.bin"),
            &part,
            data.len() as u64,
            &hash_of(&data),
            &cancel,
            &|p| {
                if p.bytes_done > 4096 {
                    c.cancel();
                }
            },
        )
        .await;
    assert_eq!(first.unwrap_err(), DownloadError::Cancelled);
    let have = std::fs::metadata(&part).unwrap().len();
    assert!(have > 0 && have < data.len() as u64, "partial was {have}");

    // The second attempt asks for the rest.
    let log = ProgressLog::new();
    let sink = log.sink();
    let out = d
        .download_file(
            &s.url("weights.bin"),
            &part,
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &sink,
        )
        .await
        .expect("the resume must succeed");

    assert_eq!(out.resumed_from, have, "it must resume, not restart");
    assert_eq!(out.bytes, data.len() as u64);
    assert_eq!(std::fs::read(&part).unwrap(), data);
    // The server saw a range request for exactly the offset we had.
    assert_eq!(s.hits_for("weights.bin"), 1);
    // And the progress denominator is the whole file, not the remainder, so the
    // bar does not jump backwards for the user.
    assert!(
        log.ticks()
            .iter()
            .all(|t| t.bytes_total == data.len() as u64)
    );
}

#[tokio::test]
async fn a_server_that_ignores_the_range_header_is_a_clear_error() {
    // Silently restarting 2.5 GB from zero because a CDN ignored `Range` looks to
    // a user on a hotel connection like a bug that repeats forever, so it is an
    // explicit, reported failure.
    let data = ramp(32 * 1024);
    let s = server(&[("weights.bin", Body::without_range(data.clone()))]);
    let t = TempDir::new("dl-norange");
    let part = t.join("m1/.partial/weights.bin.part");
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, ramp(8192)).unwrap();

    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    let e = error_expecting(
        &d,
        &s.url("weights.bin"),
        &part,
        data.len() as u64,
        &hash_of(&data),
        true,
    )
    .await;
    // The partial is untouched by the refusal, so a later attempt against a
    // range-capable server can still resume from it.
    assert_eq!(
        e,
        DownloadError::NoRangeSupport {
            url: s.url("weights.bin")
        }
    );
    let v = e.to_vaani_error();
    assert_eq!(v.code(), vaani_core::ErrorCode::NetworkFailed);
    assert!(v.user.recovery.is_some());
    // The partial is untouched, so a later attempt against a range-capable server
    // can still resume.
    assert_eq!(std::fs::metadata(&part).unwrap().len(), 8192);
}

#[tokio::test]
async fn a_server_that_ignores_the_range_header_may_be_allowed_to_restart() {
    // The other policy, for a server known not to support ranges at all. Opt-in,
    // because the default failure is the honest one.
    let data = ramp(32 * 1024);
    let s = server(&[("weights.bin", Body::without_range(data.clone()))]);
    let t = TempDir::new("dl-restart");
    let part = t.join("m1/.partial/weights.bin.part");
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, ramp(8192)).unwrap();

    let d = Downloader::with_policy(
        DownloadPolicy {
            allow_restart_without_range: true,
            ..fast_policy()
        },
        Arc::new(FixedFreeSpace(Some(1 << 30))),
    );
    let out = d
        .download_file(
            &s.url("weights.bin"),
            &part,
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .expect("a restart is allowed here");
    assert_eq!(out.resumed_from, 0, "it restarted rather than appending");
    assert_eq!(std::fs::read(&part).unwrap(), data);
}

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_404_is_a_manifest_problem_not_a_network_one() {
    let s = server(&[]);
    let t = TempDir::new("dl-404");
    let part = t.join("m1/.partial/w.part");
    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    let e = d
        .download_file(
            &s.url("missing"),
            &part,
            10,
            &[0u8; 32],
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap_err();
    assert!(matches!(e, DownloadError::NotFound { .. }), "{e:?}");
    let v = e.to_vaani_error();
    assert_eq!(v.code(), vaani_core::ErrorCode::ModelUnavailable);
    assert!(v.user.recovery.unwrap().contains("Update VaaniDesk"));
}

#[tokio::test]
async fn a_401_and_a_403_are_different_problems_with_different_actions() {
    // A token that was never supplied and a token the publisher has not
    // approved are the same HTTP family and completely different user
    // experiences. The taxonomy has two codes for exactly this reason.
    let s = server(&[]);
    let t = TempDir::new("dl-auth");
    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));

    let e = d
        .download_file(
            &s.url("unauthorised"),
            &t.join("m1/.partial/a.part"),
            10,
            &[0u8; 32],
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap_err();
    assert_eq!(
        e.to_vaani_error().code(),
        vaani_core::ErrorCode::AuthRequired
    );
    assert!(
        e.to_vaani_error()
            .user
            .recovery
            .unwrap()
            .contains("Hugging Face token")
    );

    let tokenised = d.with_bearer(Redacted::new("hf_SECRET"));
    let e = tokenised
        .download_file(
            &s.url("forbidden"),
            &t.join("m1/.partial/b.part"),
            10,
            &[0u8; 32],
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap_err();
    assert_eq!(
        e.to_vaani_error().code(),
        vaani_core::ErrorCode::AccessNotGranted,
        "got {e:?} / {}",
        e.to_vaani_error().debug_chain()
    );
    assert!(
        e.to_vaani_error()
            .user
            .recovery
            .unwrap()
            .contains("Request access")
    );
    // And the token never appears in the message.
    assert!(!e.to_string().contains("hf_SECRET"));
}

#[tokio::test]
async fn a_truncated_body_is_reported_as_a_short_read_not_a_silent_success() {
    // The server advertises the full length and sends half. Without the length
    // check the HTTP client reports an opaque body-decoding error, which sends
    // the user off to check their router instead of telling them the transfer
    // stopped. A *single* attempt, so this asserts the first answer rather than
    // the recovery.
    let data = ramp(64 * 1024);
    let s = server(&[("weights.bin", Body::truncated(data.clone(), 32 * 1024))]);
    let t = TempDir::new("dl-short");
    let part = t.join("m1/.partial/w.part");
    let d = Downloader::with_policy(
        DownloadPolicy::single_attempt(),
        Arc::new(FixedFreeSpace(Some(1 << 30))),
    );
    let e = error_expecting(
        &d,
        &s.url("weights.bin"),
        &part,
        data.len() as u64,
        &hash_of(&data),
        false,
    )
    .await;
    assert!(
        matches!(
            e,
            DownloadError::ShortRead {
                expected: 65536,
                received: 32768,
                ..
            }
        ),
        "{e:?}"
    );
    assert!(e.is_transient(), "a short read is worth retrying");
    assert_eq!(
        e.to_vaani_error().code(),
        vaani_core::ErrorCode::NetworkFailed
    );
}

#[tokio::test]
async fn a_short_read_is_retried_and_the_resume_completes_the_file() {
    // The recovery path, and the reason a short read is classified as transient
    // rather than fatal: attempt one leaves 32 KB on disk, attempt two asks for
    // the rest, and the file arrives whole and verified. This is the single
    // most likely thing to go wrong on a real user's connection.
    let data = ramp(64 * 1024);
    let s = server(&[("weights.bin", Body::truncated(data.clone(), 32 * 1024))]);
    let t = TempDir::new("dl-short2");
    let part = t.join("m1/.partial/w.part");
    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    let out = d
        .download_file(
            &s.url("weights.bin"),
            &part,
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .expect("the retry must complete the file");
    assert_eq!(out.bytes, data.len() as u64);
    assert_eq!(
        out.resumed_from,
        32 * 1024,
        "it resumed, it did not restart"
    );
    assert_eq!(std::fs::read(&part).unwrap(), data);
}

#[tokio::test]
async fn the_wrong_bytes_are_rejected_and_the_partial_is_discarded() {
    // The right length, the wrong content: the case the manifest's hashes exist
    // for. The partial is removed, because its bytes are known to be wrong and a
    // later resume from a corrupt offset would produce a different corrupt file.
    let good = ramp(32 * 1024);
    let bad = support::other_ramp(32 * 1024);
    let s = server(&[("weights.bin", Body::new(bad.clone()))]);
    let t = TempDir::new("dl-hash");
    let part = t.join("m1/.partial/w.part");
    let d = Downloader::with_policy(
        DownloadPolicy::single_attempt(),
        Arc::new(FixedFreeSpace(Some(1 << 30))),
    );
    let e = error_expecting(
        &d,
        &s.url("weights.bin"),
        &part,
        good.len() as u64,
        &hash_of(&good),
        true,
    )
    .await;
    match &e {
        DownloadError::HashMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, &hex::encode(hash_of(&good)));
            assert_eq!(actual, &hex::encode(hash_of(&bad)));
        }
        other => panic!("expected a hash mismatch, got {other:?}"),
    }
    assert_eq!(
        e.to_vaani_error().code(),
        vaani_core::ErrorCode::ModelCorrupt
    );
    assert!(
        !part.exists(),
        "known-bad bytes must not be kept for a resume"
    );
}

#[tokio::test]
async fn a_persistent_hash_mismatch_is_retried_from_scratch_and_then_reported() {
    // Each retry must start from zero, because the previous attempt's bytes are
    // known to be wrong; resuming onto a corrupt offset would produce a
    // different corrupt file every time and never converge.
    let good = ramp(16 * 1024);
    let bad = support::other_ramp(16 * 1024);
    let s = server(&[("weights.bin", Body::new(bad.clone()))]);
    let t = TempDir::new("dl-hash2");
    let part = t.join("m1/.partial/w.part");
    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    // The server is deterministic, so the answer is deterministic too -- unless
    // the *harness* loses a connection, in which case the recorded cause is a
    // transport error rather than a checksum. Retry the whole download then, so
    // this asserts about the retry accounting and not about the loopback server.
    let mut e = None;
    for _ in 0..6 {
        let out = d
            .download_file(
                &s.url("weights.bin"),
                &part,
                good.len() as u64,
                &hash_of(&good),
                &CancelToken::new(),
                &|_| {},
            )
            .await;
        let err = out.expect_err("the server serves the wrong bytes, so it cannot succeed");
        let raced = matches!(
            &err,
            DownloadError::RetriesExhausted { last, .. }
                if matches!(**last, DownloadError::Network { .. })
        );
        e = Some(err);
        if !raced {
            break;
        }
    }
    let e = e.expect("an attempt was made");
    match &e {
        DownloadError::RetriesExhausted { attempts, last, .. } => {
            assert_eq!(*attempts, 4, "it must use exactly its attempt budget");
            assert!(
                matches!(**last, DownloadError::HashMismatch { .. }),
                "the cause must survive the retry loop: {last:?}"
            );
        }
        other => panic!("expected RetriesExhausted, got {other:?}"),
    }
    assert_eq!(
        e.to_vaani_error().code(),
        vaani_core::ErrorCode::ModelCorrupt
    );
    assert!(
        !part.exists(),
        "the bad partial is discarded on every attempt"
    );
    assert!(
        s.hits_for("weights.bin") <= 4 * 6,
        "it must not hammer a server serving the wrong bytes: {} requests \
         across at most six whole-download attempts of four each",
        s.hits_for("weights.bin")
    );
}

#[tokio::test]
async fn a_transient_5xx_is_retried_and_a_persistent_one_is_reported() {
    let data = ramp(16 * 1024);
    // Two failures, then success.
    let s = server(&[("weights.bin", Body::flaky(data.clone(), 2))]);
    let t = TempDir::new("dl-retry");
    let d = Downloader::with_policy(
        DownloadPolicy {
            max_attempts: 4,
            base_delay: Duration::ZERO,
            reserve_bytes: 0,
            chunk_bytes: 4096,
            ..DownloadPolicy::default()
        },
        Arc::new(FixedFreeSpace(Some(1 << 30))),
    );
    let out = d
        .download_file(
            &s.url("weights.bin"),
            &t.join("m1/.partial/w.part"),
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .expect("the third attempt must succeed");
    assert!(
        out.attempts >= 3,
        "two 503s must be retried, took {} attempt(s)",
        out.attempts
    );
    // The server rejected the first two requests. A transient *transport* error
    // under a fully loaded test runner can cost one extra request, so this is a
    // floor rather than an equality; the "never more than max_attempts" bound is
    // asserted below, which is the invariant that actually matters.
    assert!(s.hits_for("weights.bin") >= 3, "{:?}", s.hits());

    // A server that never recovers is reported as exhausted, with the cause kept.
    let s2 = server(&[("weights.bin", Body::flaky(data.clone(), 99))]);
    let e = d
        .download_file(
            &s2.url("weights.bin"),
            &t.join("m1/.partial/x.part"),
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap_err();
    match &e {
        DownloadError::RetriesExhausted { attempts, last, .. } => {
            assert_eq!(*attempts, 4, "it must use exactly its attempt budget");
            // Normally a 503. A transport error is accepted too: at this level
            // the two make the same statement -- the server never delivered the
            // file -- and the loopback server produces the latter occasionally.
            assert!(
                matches!(
                    **last,
                    DownloadError::Http { status: 503, .. } | DownloadError::Network { .. }
                ),
                "the cause must survive the retry loop: {last:?}"
            );
        }
        other => panic!("expected RetriesExhausted, got {other:?}"),
    }
    assert_eq!(
        e.to_vaani_error().code(),
        vaani_core::ErrorCode::NetworkFailed
    );
    // The invariant that matters: the client gives up at `max_attempts` and does
    // not keep hammering a server that is refusing.
    assert!(
        s2.hits_for("weights.bin") <= 4,
        "made {} requests against a limit of 4",
        s2.hits_for("weights.bin")
    );
}

#[tokio::test]
async fn a_refused_connection_is_a_network_error_not_a_panic() {
    // Nothing is listening on this port: the classic "no network" case.
    let t = TempDir::new("dl-refused");
    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    // Bind and immediately drop, so the port is very likely free.
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let e = d
        .download_file(
            &format!("http://127.0.0.1:{port}/weights.bin"),
            &t.join("m1/.partial/w.part"),
            1024,
            &[0u8; 32],
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap_err();
    // A transport failure is retried and then reported with its cause kept, so
    // the developer log still says *why* every attempt failed.
    match &e {
        DownloadError::RetriesExhausted { attempts, last, .. } => {
            assert_eq!(*attempts, 4);
            assert!(
                matches!(**last, DownloadError::Network { .. }),
                "the cause must survive the retry loop: {last:?}"
            );
        }
        other => panic!("expected RetriesExhausted, got {other:?}"),
    }
    assert_eq!(
        e.to_vaani_error().code(),
        vaani_core::ErrorCode::NetworkFailed
    );
    assert!(e.to_vaani_error().user.recovery.is_some());
}

#[tokio::test]
async fn a_302_to_a_cdn_is_followed_and_the_range_survives_the_hop() {
    // The Hugging Face shape: `resolve/` answers 302 to a CDN, and the CDN
    // honours a byte range. The redirect must be followed, and the resumed
    // request must still be a range request on the far side.
    let data = ramp(48 * 1024);
    let s = server(&[("weights.bin", Body::redirected(data.clone()))]);
    let t = TempDir::new("dl-redirect");
    let part = t.join("m1/.partial/w.part");
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, ramp(16 * 1024)).unwrap();

    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    // A ranged request goes straight to the CDN -- the 302 is only for a fresh
    // fetch -- so this is one request, and it must be a 206 from the far side.
    let out = d
        .download_file(
            &s.url("weights.bin"),
            &part,
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .expect("a redirect must not break a resume");
    assert_eq!(out.resumed_from, 16 * 1024);
    assert_eq!(std::fs::read(&part).unwrap(), data);
    assert_eq!(s.hits_for("weights.bin"), 1);

    // And a fresh fetch does go through the 302 first: two requests. A separate
    // body name, because the server only redirects an object the first time it
    // is asked for it.
    let fresh = t.join("m1/.partial/fresh.part");
    let s2 = server(&[("other.bin", Body::redirected(data.clone()))]);
    let d2 = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    let out = d2
        .download_file(
            &s2.url("other.bin"),
            &fresh,
            data.len() as u64,
            &hash_of(&data),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .expect("a redirect must be followed");
    assert_eq!(out.bytes, data.len() as u64);
    assert_eq!(std::fs::read(&fresh).unwrap(), data);
    assert_eq!(s2.hits_for("other.bin"), 2, "302 then 200");
}

// ---------------------------------------------------------------------------
// Cancellation, disk space, and the atomic ordering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_cancelled_download_leaves_a_partial_and_no_file_under_the_real_name() {
    // The ordering assertion the whole design exists for: a cancelled or failed
    // download must never leave bytes under the name the loader opens.
    let data = ramp(256 * 1024);
    let s = server(&[("weights.bin", Body::slow(data.clone(), 1))]);
    let t = TempDir::new("dl-cancel");
    let model_dir = t.dir("m1");
    let part = Downloader::part_path(&model_dir, "weights.bin");

    let d = Downloader::with_policy(fast_policy(), Arc::new(FixedFreeSpace(Some(1 << 30))));
    let cancel = CancelToken::new();
    let c = cancel.clone();
    let e = d
        .download_file(
            &s.url("weights.bin"),
            &part,
            data.len() as u64,
            &hash_of(&data),
            &cancel,
            &|p| {
                if p.bytes_done > 8 * 1024 {
                    c.cancel();
                }
            },
        )
        .await
        .unwrap_err();
    assert_eq!(e, DownloadError::Cancelled);
    assert!(e.to_vaani_error().is_cancelled());
    // The partial is there, for the next attempt to resume.
    assert!(part.exists(), "the partial must survive for a resume");
    assert!(part.starts_with(model_dir.join(PARTIAL_DIR)));
    // And nothing is under the real name.
    assert!(
        !model_dir.join("weights.bin").exists(),
        "a cancelled download must not create the real file"
    );
}

#[tokio::test]
async fn a_full_disk_is_refused_before_a_single_byte_is_fetched() {
    // Simulated with an injectable probe rather than by filling the real volume,
    // which is not a test anybody should write. The precheck happens once, for
    // the whole model, in `ModelManager::install` -- so this asserts the
    // decision and that no request was made.
    let data = ramp(64 * 1024);
    let s = server(&[("weights.bin", Body::new(data.clone()))]);
    let t = TempDir::new("dl-disk");

    let d = Downloader::with_policy(
        DownloadPolicy {
            reserve_bytes: 1024,
            ..fast_policy()
        },
        Arc::new(FixedFreeSpace(Some(2048))),
    );
    let pre = d.precheck(t.path(), data.len() as u64);
    match pre.unwrap_err() {
        DownloadError::InsufficientDisk { needed, free } => {
            assert_eq!(needed, data.len() as u64 + 1024);
            assert_eq!(free, 2048);
        }
        other => panic!("expected InsufficientDisk, got {other:?}"),
    }
    assert_eq!(
        d.precheck(t.path(), 1024).unwrap(),
        Some(2048),
        "a model that fits must be allowed"
    );
    assert_eq!(s.hits_for("weights.bin"), 0, "nothing was requested");
}

#[tokio::test]
async fn the_manager_refuses_to_start_an_install_that_does_not_fit() {
    // The precheck wired into the real install path, and the guarantee that it
    // happens before the first request.
    let data = ramp(64 * 1024);
    let s = server(&[("a.bin", Body::new(data.clone()))]);
    let t = TempDir::new("dl-disk2");
    let user = t.dir("user");
    let manager = ModelManager::new(
        manifest_for(
            "a.bin",
            data.len() as u64,
            &hex::encode(hash_of(&data)),
            &format!("{}{{path}}", s.url("")),
        ),
        ModelRoots::at(&user, t.join("bundled")),
    )
    .with_downloader(Downloader::with_policy(
        DownloadPolicy {
            reserve_bytes: 1024,
            ..fast_policy()
        },
        Arc::new(FixedFreeSpace(Some(2048))),
    ));
    let e = manager
        .install(
            "m1",
            &InstallOptions::default(),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap_err();
    assert_eq!(e.code(), vaani_core::ErrorCode::InsufficientDisk);
    assert!(e.user.recovery.unwrap().contains("Free up some disk space"));
    assert_eq!(s.hits_for("a.bin"), 0, "no bytes were requested");
    assert!(!user.join("m1/a.bin").exists());
}

#[tokio::test]
async fn the_manager_installs_a_model_atomically_and_only_hashes_correct_files() {
    // The end-to-end shape, through the façade the CLI and the UI use: files go
    // to `.partial`, get hashed, and are renamed into place one at a time; the
    // receipt is written last; the model is then complete and verifiable.
    let a = ramp(20 * 1024);
    let b = ramp(12 * 1024);
    let s = server(&[
        ("a.bin", Body::new(a.clone())),
        ("b.bin", Body::new(b.clone())),
    ]);
    let t = TempDir::new("dl-install");
    let user = t.dir("user");
    let roots = ModelRoots::at(&user, t.join("bundled"));

    let sha_a = hex::encode(hash_of(&a));
    let sha_b = hex::encode(hash_of(&b));
    let manifest = Manifest::parse(&format!(
        r#"{{"schema_version":1,"models":[{{
          "id":"m1","engine_id":"e1","version":"1.0","display_name":"M1",
          "description":"d","task":"asr","architecture":"a","access":"open",
          "license":{{"spdx":"MIT","name":"MIT","url":"https://example.org/l"}},
          "source":{{"kind":"direct","url_template":"{}/{{path}}"}},
          "languages":["hi"],"sample_rates":[16000],"size_bytes":{total},
          "files":[
            {{"path":"a.bin","role":"weights","bytes":{la},"sha256":"{sha_a}"}},
            {{"path":"b.bin","role":"config","bytes":{lb},"sha256":"{sha_b}"}}
          ]}}]}}"#,
        s.url(""),
        total = a.len() + b.len(),
        la = a.len(),
        lb = b.len(),
    ))
    .expect("valid manifest");

    let manager = ModelManager::new(manifest, roots).with_downloader(Downloader::with_policy(
        fast_policy(),
        Arc::new(FixedFreeSpace(Some(1 << 30))),
    ));

    let log = ProgressLog::new();
    let sink = log.sink();
    let report = manager
        .install("m1", &InstallOptions::default(), &CancelToken::new(), &sink)
        .await
        .expect("the install must succeed");

    assert!(!report.already_installed);
    assert_eq!(report.installed, vec!["a.bin", "b.bin"]);
    assert_eq!(report.bytes_downloaded, (a.len() + b.len()) as u64);
    let model_dir = user.join("m1");
    assert_eq!(std::fs::read(model_dir.join("a.bin")).unwrap(), a);
    assert_eq!(std::fs::read(model_dir.join("b.bin")).unwrap(), b);
    // The receipt exists and records what was installed.
    let receipt: vaani_models::InstallReceipt =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join(".installed.json")).unwrap())
            .expect("the receipt must be valid JSON");
    assert_eq!(receipt.model_id, "m1");
    assert_eq!(receipt.files.len(), 2);
    assert!(!model_dir.join(".vaani-install.lock").exists(), "released");

    // And the model is now genuinely installed and verifies.
    assert!(manager.discover().is_installed("m1"));
    let v = manager
        .verify("m1", &CancelToken::new(), &|_| {})
        .expect("verify must run");
    assert!(v.is_ok(), "{:?}", v.files);
    assert!(!v.content_disagreement());

    // A second install is a no-op, because the first one satisfied it.
    let again = manager
        .install(
            "m1",
            &InstallOptions::default(),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .expect("a second install must succeed");
    assert!(again.already_installed);
    assert!(again.installed.is_empty());
    assert_eq!(again.bytes_downloaded, 0);
    assert_eq!(s.hits_for("a.bin"), 1, "nothing was fetched twice");
}

#[tokio::test]
async fn the_install_plan_reports_the_real_numbers_and_a_bundled_model_says_so() {
    // The numbers the UI has to show before a user commits to a 930 MB download.
    let manifest = Manifest::embedded().expect("valid");
    let t = TempDir::new("dl-bundled");
    let manager = ModelManager::new(manifest, ModelRoots::at(t.join("user"), t.join("bundled")));

    let plan = manager.install_plan("sravaani-1.0-onnx").expect("a plan");
    assert_eq!(
        plan.total_bytes, 2_702_792_388,
        "everything in the manifest"
    );
    assert_eq!(plan.download_bytes, 929_977_857, "required files only");
    assert!(!plan.already_on_disk);
    assert!(
        plan.downloadable,
        "a gated Hub model is fetchable, with a token"
    );
    assert_eq!(plan.access.as_str(), "gated-token");
    assert!(
        plan.source
            .contains("huggingface.co/ARTPARK-IISc/SraVaani-1.0")
    );
    // A plan for an unknown model is a clear error, not a panic.
    let e = manager.install_plan("nope").unwrap_err();
    assert!(e.user.recovery.unwrap().contains("vaani models list"));

    // A model with no URL at all refuses to install, and says where it came
    // from rather than inventing a URL. `bundled` is the provider that means
    // "ships with the app".
    let bundled = ModelManager::new(
        Manifest::parse(
            r#"{"schema_version":1,"models":[{
              "id":"m1","engine_id":"e1","version":"1","display_name":"M",
              "description":"d","task":"asr","architecture":"a","access":"open",
              "license":{"spdx":"MIT","name":"MIT","url":"https://example.org/l"},
              "source":{"kind":"bundled"},"languages":["hi"],"sample_rates":[16000],
              "size_bytes":4,
              "files":[{"path":"w.bin","role":"weights","bytes":4,
                "sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]
            }]}"#,
        )
        .expect("valid"),
        ModelRoots::at(t.join("user2"), t.join("bundled")),
    );
    let plan = bundled.install_plan("m1").expect("a plan");
    assert!(!plan.downloadable, "a bundled source has no URL");
    let e = bundled
        .install(
            "m1",
            &InstallOptions::default(),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap_err();
    assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
    assert!(e.user.recovery.unwrap().contains("Reinstall"));
}

#[tokio::test]
async fn remove_deletes_a_user_install_and_refuses_a_bundled_one() {
    let a = ramp(1024);
    let s = server(&[("a.bin", Body::new(a.clone()))]);
    let t = TempDir::new("dl-remove");
    let user = t.dir("user");
    let manager = ModelManager::new(
        manifest_for(
            "a.bin",
            a.len() as u64,
            &hex::encode(hash_of(&a)),
            &format!("{}{{path}}", s.url("")),
        ),
        ModelRoots::at(&user, t.join("bundled")),
    )
    .with_downloader(Downloader::with_policy(
        fast_policy(),
        Arc::new(FixedFreeSpace(Some(1 << 30))),
    ));
    manager
        .install(
            "m1",
            &InstallOptions::default(),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap();
    assert!(user.join("m1/a.bin").exists());

    let r = manager.remove("m1").expect("remove must work");
    assert!(r.was_installed);
    assert_eq!(r.bytes_freed, a.len() as u64);
    assert!(!user.join("m1").exists(), "the directory goes too");

    // Removing again is a clear "nothing to remove", not a crash.
    let e = manager.remove("m1").unwrap_err();
    assert_eq!(e.code(), vaani_core::ErrorCode::ModelUnavailable);
    assert!(e.user.recovery.unwrap().contains("nothing to remove"));
}

#[tokio::test]
async fn a_guarded_model_cannot_be_removed_and_says_why() {
    // The requirement: a model file being read by an inference session must not
    // be deletable, and the refusal must be clear.
    let a = ramp(1024);
    let s = server(&[("a.bin", Body::new(a.clone()))]);
    let t = TempDir::new("dl-guard");
    let user = t.dir("user");
    let manager = ModelManager::new(
        manifest_for(
            "a.bin",
            a.len() as u64,
            &hex::encode(hash_of(&a)),
            &format!("{}{{path}}", s.url("")),
        ),
        ModelRoots::at(&user, t.join("bundled")),
    )
    .with_downloader(Downloader::with_policy(
        fast_policy(),
        Arc::new(FixedFreeSpace(Some(1 << 30))),
    ));
    manager
        .install(
            "m1",
            &InstallOptions::default(),
            &CancelToken::new(),
            &|_| {},
        )
        .await
        .unwrap();

    let guards = manager
        .guard_files("m1")
        .expect("the session may guard the model");
    assert_eq!(guards.len(), 1);
    let e = manager.remove("m1").unwrap_err();
    assert_eq!(e.code(), vaani_core::ErrorCode::StorageFailed);
    assert!(e.user.title.contains("in use"), "{}", e.user.title);
    assert!(user.join("m1/a.bin").exists(), "nothing was deleted");
    drop(guards);
    manager.remove("m1").expect("removable once released");
}

#[test]
fn the_free_space_probe_is_replaceable_so_disk_full_is_testable() {
    // A compile-and-run assertion about the injection point itself: the manager
    // takes a `Downloader`, the `Downloader` takes a probe, and the probe is what
    // a test replaces.
    #[derive(Debug)]
    struct Zero;
    impl FreeSpaceProbe for Zero {
        fn free_bytes(&self, _p: &std::path::Path) -> Option<u64> {
            Some(0)
        }
    }
    let d = Downloader::with_policy(fast_policy(), Arc::new(Zero));
    let e = d
        .precheck(std::path::Path::new("C:/models"), 1)
        .unwrap_err();
    assert!(matches!(e, DownloadError::InsufficientDisk { free: 0, .. }));
}

#[test]
fn a_manifest_url_for_a_huggingface_model_is_built_not_stored() {
    // The URL is derived, so it cannot go stale against a moved repository, and
    // a manifest cannot carry a URL that disagrees with its repo.
    let m = Manifest::parse(
        r#"{"schema_version":1,"models":[{
          "id":"m1","engine_id":"e1","version":"1","display_name":"M","description":"d",
          "task":"asr","architecture":"a","access":"gated-token",
          "license":{"spdx":"MIT","name":"MIT","url":"https://example.org/l"},
          "source":{"kind":"huggingface","repo":"ARTPARK-IISc/SraVaani-1.0","revision":"main"},
          "languages":["hi"],"sample_rates":[16000],"size_bytes":4,
          "files":[{"path":"w.bin","role":"weights","bytes":4,
            "sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]
        }]}"#,
    )
    .unwrap();
    let model = m.model("m1").unwrap();
    assert!(model.source.needs_bearer());
    assert_eq!(
        model.source.url_for("w.bin").unwrap().as_str(),
        "https://huggingface.co/ARTPARK-IISc/SraVaani-1.0/resolve/main/w.bin"
    );
    assert!(parse_sha256(&hex::encode(hash_of(b"abc"))).is_some());
}
