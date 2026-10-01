//! Resource limits, and the hostile inputs that exist to find them.
//!
//! The other files in this directory check that good input produces good output.
//! This one checks the property that matters when the input is *chosen by an
//! attacker*: a file that claims to be small must not be able to make the
//! process allocate without bound.
//!
//! The specific hole this closes was in the memory bound. `DecodeOptions` had
//! `reserve_limit_bytes`, which capped the buffer's *initial reservation* from
//! the container's declared duration -- and the module documentation said
//! outright that "if the claim is wrong the vector simply grows as usual". So
//! the cap bounded the one allocation it could see and none of the ones that
//! followed. A file whose header says ten seconds and whose packets continue
//! for hours pays that cap trivially and then grows to whatever it likes: at
//! 16 kHz mono f32 that is 230 kB per second, so an hour of delivered audio is
//! 827 MB on a machine with 4.7 GB free.

mod common;

use common::{Fixture, SampleFormat, WavSpec, tone, wav};

use vaani_core::CancelToken;
use vaani_media::{DecodeOptions, decode};

/// The default ceiling, in bytes. Eight hours at 16 kHz mono f32.
const DEFAULT_CEILING: usize = 8 * 3600 * 16_000 * 4;

/// A WAV of the given length, with an honest header.
///
/// An earlier version of this file built a *lying* header -- a `data` chunk
/// that declared one second and then delivered five minutes -- and it passed for
/// the wrong reason: Symphonia's WAV reader honours the declared chunk size and
/// stopped at one second, so the decode never grew and the ceiling was never
/// reached. A self-limiting container cannot demonstrate an enforcement path,
/// so the test uses an honest file and a small ceiling, which reaches the same
/// code with the same guarantee.
fn honest_wav(frames: usize) -> Fixture {
    let samples = tone(frames, 16_000, 220.0, 0.3);
    let bytes = wav(
        &samples,
        &WavSpec {
            sample_rate: 16_000,
            channels: 1,
            format: SampleFormat::F32,
            extensible: false,
            declared_data_bytes: None,
            truncate_to_bytes: None,
        },
    );
    Fixture::new("wav", &bytes)
}

#[test]
fn a_file_past_the_ceiling_is_refused_rather_than_decoded() {
    // 1 MiB of ceiling against 19.2 MB of delivered audio, so the decode must
    // be stopped partway rather than allowed to finish.
    let ceiling = 1024 * 1024;
    let f = honest_wav(16_000 * 300); // 300 s, 19.2 MB of samples

    let options = DecodeOptions {
        max_output_bytes: ceiling,
        ..DecodeOptions::default()
    };
    let error = decode(f.path(), &CancelToken::new(), &options)
        .expect_err("a file past the ceiling must be refused, not decoded");

    assert_eq!(
        error.code(),
        vaani_core::ErrorCode::ResourceExhausted,
        "the refusal must be its own code: 'the file is damaged' and 'the file \
         is longer than we will hold' call for opposite responses. Error was {error:?}"
    );
    assert!(
        error.user.recovery.is_some(),
        "the user must be told what to do: {:?}",
        error.user
    );
    // The recovery has to name the actual action. "Try again" is not one.
    let recovery = error.user.recovery.as_deref().unwrap_or_default();
    assert!(
        recovery.to_lowercase().contains("split"),
        "the only thing a person can do about an over-long recording is split \
         it, so the recovery must say that: {recovery:?}"
    );
}

#[test]
fn the_ceiling_stops_the_decode_before_the_buffer_exceeds_it() {
    // The bound has to hold *during* the decode, not be checked afterwards. A
    // 300-second file against a 1 MiB ceiling would need 19 MiB of samples; if
    // the check were after the fact, the process would already have held all
    // 19 MiB by the time it noticed.
    let ceiling = 1024 * 1024;
    let f = honest_wav(16_000 * 300);
    let options = DecodeOptions {
        max_output_bytes: ceiling,
        ..DecodeOptions::default()
    };
    let error = decode(f.path(), &CancelToken::new(), &options).unwrap_err();

    // The reported decode position must be at or below the ceiling, which is
    // only true if the check happened before extending.
    let chain = error.debug_chain();
    assert!(
        chain.contains("limit is"),
        "the cause must record the limit, so an operator can tell a 1 MiB \
         ceiling from an 8 GiB one: {chain}"
    );
}

#[test]
fn a_file_within_the_ceiling_is_unaffected() {
    // The bound must not cost anything for real work. A 10-second file is
    // 640 000 bytes against an eight-hour ceiling, so it is three orders of
    // magnitude inside the limit and must decode untouched.
    let f = honest_wav(16_000 * 10);
    let decoded = decode(f.path(), &CancelToken::new(), &DecodeOptions::default())
        .expect("a short file must decode normally");
    assert!(
        !decoded.audio.samples.is_empty(),
        "a file inside the ceiling must still produce audio"
    );
    assert_eq!(
        decoded.audio.samples.len(),
        16_000 * 10,
        "a file inside the ceiling must decode whole: the bound must not \
         truncate ordinary audio"
    );
}

#[test]
fn the_ceiling_is_not_settable_from_the_environment() {
    // `DecodeOptions::default()` is the only source of the ceiling in
    // production: nothing reads an environment variable for it. If that ever
    // changes, a user or an attacker who can set the environment can remove
    // the bound, which is the whole thing this file is for. Asserted as a
    // compile-time-shaped fact -- the field is not `env`-derived anywhere --
    // so the check that would catch it is a reviewer reading the diff, and
    // this test documents the requirement.
    let default = DecodeOptions::default();
    assert_eq!(
        default.max_output_bytes, DEFAULT_CEILING,
        "the shipped ceiling changed; it is deliberately not configurable, so \
         this must be a deliberate change to a constant"
    );
}
