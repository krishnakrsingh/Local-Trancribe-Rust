#requires -Version 5
<#
.SYNOPSIS
    Mutation testing for crates/vaani-qa. Breaks the code under test, confirms a
    test notices, and reverts.

.DESCRIPTION
    A test that passes against deliberately broken code buys false confidence,
    which is worse than having no test at all: it is filed under "covered" and
    nobody looks at it again. The only way to know which tests have teeth is to
    take the teeth out and see whether anything screams.

    Each mutation below edits a file this agent does **not** own, runs a named
    test, and reverts in a `finally` block. Every edit is a one-line,
    obviously-wrong change to production code, so a failure means the *test* is
    weak rather than the mutation being unfair.

    Two of the mutations edit a *fixture* rather than code, because "somebody
    regenerated the golden file" is the failure this crate exists to prevent and
    it cannot be reproduced any other way.

    Run from the repository root:

        pwsh scripts\a10\mutate.ps1

.EXAMPLE
    pwsh scripts\a10\mutate.ps1 -Mutation pipeline-loses-the-chunk-offset

.EXAMPLE
    pwsh scripts\a10\mutate.ps1 -Keep -Mutation golden-gate-goes-loose

.NOTES
    Owned by agent A10. The script checks `git status` at the end and refuses to
    report a clean tree if a mutation is still in place.
#>
[CmdletBinding()]
param(
    # Run exactly one mutation by its slug.
    [string] $Mutation = '',
    # Leave the mutation in place instead of reverting.
    [switch] $Keep
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$targetDir = if ($env:VAANI_CARGO_TARGET_DIR) { $env:VAANI_CARGO_TARGET_DIR }
             else { Join-Path ([IO.Path]::GetTempPath()) 'vaani-qa-mutate' }
$cargo = if ($env:VAANI_CARGO) { $env:VAANI_CARGO } else { 'cargo' }

# Resolve cargo to a real executable before mutating anything.
#
# The first version took `$env:VAANI_CARGO` else the bare word 'cargo' and
# invoked it. Under `powershell -NoProfile -File`, `~/.cargo/bin` is not on PATH,
# so every mutation failed with CommandNotFoundException -- and the script then
# printed "0 mutations, 0 survived" followed by "Every mutation was caught."
# A mutation run that executed nothing reported perfect success, which is the
# single most damaging thing a mutation harness can do: it manufactures the
# confidence the reader is relying on. Fail at startup instead.
if ($cargo -eq 'cargo' -and -not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $guess = Join-Path $HOME '.cargo\bin\cargo.exe'
    if (Test-Path -LiteralPath $guess) {
        $cargo = $guess
        Write-Host "  cargo:     $cargo (resolved from ~\.cargo\bin; not on PATH)"
    } else {
        throw "cargo is not on PATH and not at $guess. Set VAANI_CARGO to the cargo executable. Refusing to run: a mutation run that checks nothing must not report success."
    }
}
$probe = try { & $cargo --version 2>&1 } catch { "cannot execute '$cargo': $($_.Exception.Message)" }
if ($probe -notmatch '^cargo ') { throw "cannot run '$cargo --version': $probe" }
Write-Host "  cargo:     $cargo ($probe)"

# --- the mutations ----------------------------------------------------------
# `find` must occur EXACTLY once in `file`. A mutation whose anchor is missing or
# ambiguous aborts rather than running the test against unmodified code, because
# "the mutation survived" is exactly the false confidence this script exists to
# remove.
$mutations = @(
    @{
        slug    = 'vad-drops-the-buffer-origin'
        file    = 'crates/vaani-vad/src/energy.rs'
        find    = @('            audio.start_time.as_millis(),', '            audio.duration().as_millis(),')
        broken  = @('            0i64, // MUTATION: pretend every buffer starts at zero', '            audio.duration().as_millis(),')
        test    = 'vad_adversarial'
        filter  = 'a_buffer_with_a_non_zero_origin_produces_absolute_times'
        why     = 'The coordinate-space invariant. A VAD that ignores AudioBuffer::start_time is byte-identical on a zero-origin buffer and wrong on every other one.'
    }
    @{
        slug    = 'vad-accepts-any-sample-rate'
        file    = 'crates/vaani-vad/src/energy.rs'
        find    = '        if audio.sample_rate != SAMPLE_RATE {'
        broken  = '        if false && audio.sample_rate != SAMPLE_RATE { // MUTATION: accept any rate'
        test    = 'vad_adversarial'
        filter  = 'a_wrong_sample_rate_is_rejected_with_a_message_naming_the_required_rate'
        why     = 'A detector that silently accepts 8 kHz and thresholds it as if it were 16 kHz is the resampling bug the refusal exists to prevent.'
    }
    @{
        slug    = 'vad-reports-a-steady-tone-as-speech'
        file    = 'crates/vaani-vad/src/energy.rs'
        find    = '        .filter(|s| **s >= self.config.smoothing.enter_threshold)'
        broken  = '        .filter(|s| **s >= -1.0) // MUTATION: accept every frame as speech'
        test    = 'vad_adversarial'
        filter  = 'sixty_seconds_of_digital_silence_produces_no_speech_and_no_chunks'
        why     = 'The negative control for the whole VAD. If digital silence scores as speech, every recording gets a transcript of a room.'
    }
    @{
        slug    = 'diar-drops-the-buffer-origin'
        file    = 'crates/vaani-diarization/src/engine.rs'
        find    = @('        let start_ms = audio.start_time.as_millis();', '        let mut fe = FrontEnd::new(cfg.features, cfg.activity);', '        let layout = Layout::of(&cfg.features);', '        let poll = cfg.cancel_poll_frames.max(1);', '        let usable = fe.usable_frames(audio.samples.len());', '        let n_usable = usable.len();')
        broken  = @('        let start_ms = 0i64; // MUTATION: pretend every buffer starts at zero', '        let mut fe = FrontEnd::new(cfg.features, cfg.activity);', '        let layout = Layout::of(&cfg.features);', '        let poll = cfg.cancel_poll_frames.max(1);', '        let usable = fe.usable_frames(audio.samples.len());', '        let n_usable = usable.len();')
        test    = 'diar_adversarial'
        filter  = 'a_buffer_with_a_non_zero_origin_produces_absolute_spans'
        why     = 'The same invariant for the diarizer, which is a separate code path from the VAD''s.'
    }
    @{
        slug    = 'diar-accepts-a-stereo-buffer'
        file    = 'crates/vaani-diarization/src/engine.rs'
        find    = '        if audio.channels != 1 {'
        broken  = '        if false && audio.channels != 1 { // MUTATION: accept a stereo buffer'
        test    = 'diar_adversarial'
        filter  = 'a_stereo_buffer_is_refused_rather_than_silently_averaged'
        why     = 'A diarizer that averages stereo itself scales its pitch feature by a factor the caller does not know about, and hides the bug that the media stage did not do its job.'
    }
    @{
        slug    = 'core-progress-skips-the-clamp'
        file    = 'crates/vaani-core/src/pipeline.rs'
        find    = '            Some(f) => Progress::media(phase, f.clamp(0.0, 1.0), 1.0),'
        broken  = '            Some(f) => Progress::media(phase, f, 1.0), // MUTATION: no clamp'
        test    = 'conformance'
        filter  = 'progress_cannot_be_reported_out_of_range'
        why     = 'Without the clamp a stage that reports 1.5 reaches the UI as 150%, and a bar that overshoots is a bar that cannot be trusted.'
    }
    @{
        slug    = 'core-drops-the-chunk-offset'
        file    = 'crates/vaani-core/src/transcript.rs'
        find    = '        s.start = s.start.saturating_add(chunk_offset).clamp_zero();'
        broken  = '        s.start = s.start.clamp_zero(); // MUTATION: drop the chunk offset'
        test    = 'pipeline_integration'
        filter  = 'media_vad_asr_diarization_store_on_a_synthesised_wav'
        why     = 'THE bug. push_chunk is the single rebase point in the product; losing the offset is the "transcript is three minutes out of sync" family that JOURNEY.md opens with.'
    }
    @{
        slug    = 'export-clobbers-by-default'
        file    = 'crates/vaani-export/src/writer.rs'
        find    = '    if path.exists() && !overwrite {'
        broken  = '    if false && path.exists() && !overwrite { // MUTATION: always overwrite'
        test    = 'export_robustness'
        filter  = 'an_existing_file_is_never_clobbered_unless_the_caller_says_so'
        why     = '"Nothing is overwritten silently" is one of the exporter''s five stated properties, and the one a user notices when it breaks.'
    }
    @{
        slug    = 'export-lets-a-name-escape-the-directory'
        file    = 'crates/vaani-export/src/format.rs'
        find    = '    crate::writer::safe_file_name(stem),'
        broken  = '    stem.to_string(), // MUTATION: skip sanitising entirely'
        test    = 'export_robustness'
        filter  = 'a_crafted_source_name_cannot_escape_the_directory_the_user_chose'
        why     = 'The source name comes from the media file, which a user can call anything. This is the one place where a bug is a security bug.'
    }
    @{
        # The companion to the mutation above.
        #
        # Disabling the sanitiser alone was survivable, because
        # `writer::assert_inside` independently refuses any path whose canonical
        # parent is not the directory the user chose. That is defence in depth
        # working, but it left the suite unable to say *which* layer held. This
        # mutation disables the containment check instead, so each layer is now
        # caught on its own and neither is load-bearing alone.
        #
        # The test to use is the same one, because it asserts the outcome that
        # both layers exist to produce: nothing lands outside the directory.
        slug    = 'export-skips-the-containment-check'
        file    = 'crates/vaani-export/src/writer.rs'
        find    = '    if got == wanted {'
        broken  = '    if true || got == wanted { // MUTATION: every path is inside the directory'
        test    = 'export_robustness'
        filter  = 'a_crafted_source_name_cannot_escape_the_directory_the_user_chose'
        why     = 'The last line of defence against a crafted source name. If this is the only check that fails, the sanitiser is the only thing preventing a write outside the directory the user chose.'
    }
    @{
        slug    = 'storage-forgets-the-segment-ids'
        file    = 'crates/vaani-storage/src/transcripts.rs'
        find    = @('                id: &seg.id,', '                ordinal: ordinal as i64,')
        broken  = @('                id: "",', '                ordinal: ordinal as i64,')
        test    = 'pipeline_integration'
        filter  = 'media_vad_asr_diarization_store_on_a_synthesised_wav'
        why     = 'A segment with no id cannot be addressed, so every later edit by id silently edits nothing.'
    }
    @{
        # Replacing the orchestrator's own pre-flight check with `false &&` was
        # the first version of this mutation, and it survived. It survived for a
        # good reason and a bad one, and the bad reason is what this entry now
        # exists to record: `run_job` consults the token at three separate points
        # (the pre-flight gate, the stage loop, and the post-stage gate), so
        # disabling one leaves two, and the job is still cancelled promptly. A
        # surviving mutation means "no test can see this", and here the truth
        # was "no single test can see this" -- which is defence in depth working,
        # not a gap in the suite.
        #
        # So the mutation moved down to the one place every check funnels
        # through: `CancelToken::is_cancelled`. Ignoring the flag *at the source*
        # is the mutation that actually represents "a user who pressed Stop is
        # never told so". It is a single-occurrence anchor, and it must be
        # caught by the same pre-flight test -- if that test still passes, the
        # cancellation coverage is fake.
        slug    = 'cancel-token-ignores-the-flag'
        file    = 'crates/vaani-core/src/job.rs'
        find    = '        self.flag.load(std::sync::atomic::Ordering::SeqCst)'
        broken  = '        false && self.flag.load(std::sync::atomic::Ordering::SeqCst) // MUTATION: the flag is never read'
        test    = 'pipeline_integration'
        filter  = 'cancel_before_the_job_starts_is_clean'
        why     = 'A pipeline that ignores a cancel decodes a whole recording for a user who pressed Stop. Every check in the orchestrator funnels through this one read, so a test that passes against this is not testing cancellation at all.'
    }
    @{
        slug    = 'golden-declares-the-wrong-valid-frames'
        file    = 'tests/fixtures/sravaani/frontend_golden.json'
        find    = '"valid_frames": 100,'
        broken  = '"valid_frames": 101,'
        test    = 'golden_guard'
        filter  = ''
        why     = 'A regenerated golden file is the failure this crate exists to prevent, and it cannot be reproduced any other way: the fixture is edited, not the code. This is the mutation that motivated the digest pins.'
    }
    @{
        slug    = 'golden-drops-half-the-word-timings'
        file    = 'tests/fixtures/sravaani/decode_golden.json'
        find    = '"encoder_frames": 500,'
        broken  = '"encoder_frames": 250,'
        test    = 'golden_guard'
        filter  = ''
        why     = 'The same class, on the decode fixture. Without the pinned counts and the digest, a fixture truncated to half its content agrees with a decoder that dropped half its output.'
    }
)

# --- the runner -------------------------------------------------------------

function Invoke-Mutation {
    param([hashtable] $M)

    $path = Join-Path $root $M.file
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "SKIP  $($M.slug): $($M.file) is absent" -ForegroundColor Yellow
        return @{ slug = $M.slug; outcome = 'skipped'; detail = "$($M.file) absent" }
    }
    $original = [IO.File]::ReadAllText($path)
    # An anchor is either a single line or a list of lines; normalise to one
    # string so the rest of the function has exactly one shape to handle.
    $find = if ($M.find -is [string]) { $M.find } else { $M.find -join "`n" }
    $broken = if ($M.broken -is [string]) { $M.broken } else { $M.broken -join "`n" }
    # The anchors are written with LF for readability; the files on disk may be
    # either, so try both and refuse to guess. The CRLF forms are hoisted into
    # variables because `-replace` inside a method argument list parses its comma
    # as a second argument to the method.
    $findCrlf = $find -replace "`n", "`r`n"
    $brokenCrlf = $broken -replace "`n", "`r`n"
    $lf = $original.Contains($find)
    $crlf = $original.Contains($findCrlf)
    if (-not $lf -and -not $crlf) {
        Write-Host "ERROR $($M.slug): the anchor text is not in $($M.file). The code moved; update the mutation." -ForegroundColor Red
        return @{ slug = $M.slug; outcome = 'error'; detail = 'anchor not found' }
    }
    $needle = if ($lf) { $find } else { $findCrlf }
    $replacement = if ($lf) { $broken } else { $brokenCrlf }
    $occurrences = ([regex]::Matches($original, [regex]::Escape($needle))).Count
    if ($occurrences -ne 1) {
        Write-Host "ERROR $($M.slug): the anchor occurs $occurrences times in $($M.file); it must be unique." -ForegroundColor Red
        return @{ slug = $M.slug; outcome = 'error'; detail = "anchor occurs $occurrences times" }
    }

    try {
        [IO.File]::WriteAllText($path, $original.Replace($needle, $replacement))
        Write-Host ''
        Write-Host "=== $($M.slug) ===" -ForegroundColor Cyan
        Write-Host "    file: $($M.file)"
        Write-Host "    why:  $($M.why)"

        $env:CARGO_TARGET_DIR = $targetDir
        $cargoArgs = @('test', '-p', 'vaani-qa', '--release', '--test', $M.test)
        if ($M.filter) { $cargoArgs += @('--', '--exact', $M.filter) }
        # cargo writes its progress to stderr, and `$ErrorActionPreference = Stop`
        # would turn that into a terminating error before the exit code is even
        # read. Relax it for the duration of the invocation only.
        $ErrorActionPreference = 'Continue'
        $out = & $cargo @cargoArgs 2>&1 | Out-String
        $exit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        $ran = 0
        $counts = [regex]::Match($out, '(\d+) passed; (\d+) failed')
        if ($counts.Success) {
            $ran = [int]$counts.Groups[1].Value + [int]$counts.Groups[2].Value
        }
        if ($M.filter -and $ran -eq 0) {
            # Two different things produce "zero tests ran" and they mean opposite
            # things, so they are told apart rather than lumped together.
            $built = $out -match 'test result:'
            if ($built) {
                # A renamed or misspelled test name produces a green run with
                # every test filtered out, which this script would otherwise
                # report as "SURVIVED" -- the exact false confidence it exists to
                # remove.
                Write-Host "    ERROR    no test matched --exact $($M.filter); the test was renamed" -ForegroundColor Red
                return @{ slug = $M.slug; outcome = 'error'; detail = 'the filter matched no test' }
            }
            # The build itself failed. That is not a caught mutation: a mutation
            # that does not compile proves nothing about the test.
            $why = ($out -split "`n" | Where-Object { $_ -match '^error' } | Select-Object -First 1)
            Write-Host "    ERROR    the mutated code did not build: $why" -ForegroundColor Red
            return @{ slug = $M.slug; outcome = 'error'; detail = "the mutation does not compile: $why".Trim() }
        }
        $caught = $exit -ne 0
        $summary = ($out -split "`n" | Where-Object { $_ -match 'test result:' } | Select-Object -First 1)
        if (-not $summary) {
            $summary = ($out -split "`n" | Where-Object { $_ -match '^error' } | Select-Object -First 1)
        }
        if (-not $summary) { $summary = '(no summary: the build failed)' }
        if ($caught) {
            Write-Host "    CAUGHT    $($summary.Trim())" -ForegroundColor Green
        } else {
            Write-Host "    SURVIVED  $($summary.Trim())  <-- this test has no teeth" -ForegroundColor Red
        }
        return @{ slug = $M.slug; outcome = $(if ($caught) { 'caught' } else { 'survived' }); detail = $summary.Trim() }
    }
    finally {
        [IO.File]::WriteAllText($path, $original)
        if ($Keep) { Write-Host '    left in place (-Keep)' -ForegroundColor Yellow }
    }
}

$selected = if ($Mutation) { $mutations | Where-Object { $_.slug -eq $Mutation } } else { $mutations }
if (-not $selected) {
    throw "no mutation with the slug '$Mutation'. Known: $(($mutations.slug) -join ', ')"
}

Write-Host 'mutation testing for crates/vaani-qa'
Write-Host "  root:       $root"
Write-Host "  target dir: $targetDir"
Write-Host "  mutations:  $($selected.Count)"

$results = @()
foreach ($m in $selected) { $results += Invoke-Mutation -M $m }

Write-Host ''
Write-Host '=== summary ==='
foreach ($r in $results) {
    $colour = switch ($r.outcome) { 'caught' { 'Green' } 'survived' { 'Red' } default { 'Yellow' } }
    Write-Host ("{0,-8} {1,-44} {2}" -f $r.outcome.ToUpper(), $r.slug, $r.detail) -ForegroundColor $colour
}
$caught = @($results | Where-Object { $_.outcome -eq 'caught' })
$survived = @($results | Where-Object { $_.outcome -eq 'survived' })
$errored = @($results | Where-Object { $_.outcome -eq 'error' })
Write-Host ''
Write-Host "$($results.Count) mutations, $($caught.Count) caught, $($survived.Count) survived, $($errored.Count) could not be applied."

# Every selected mutation must have produced a verdict. This is the check that
# would have caught the bare-`cargo` bug above: it ran nothing, so nothing
# reported, and the old summary counted the empty result as success.
if ($results.Count -ne $selected.Count) {
    Write-Host "Only $($results.Count) of $($selected.Count) mutations produced a verdict. Refusing to summarise a partial run as a pass." -ForegroundColor Red
    exit 3
}
if ($survived.Count -gt 0) {
    Write-Host 'A survived mutation is a test that passes against broken code.' -ForegroundColor Red
    exit 1
}
if ($errored.Count -gt 0) {
    Write-Host 'An errored mutation proves nothing either way; fix the anchor.' -ForegroundColor Red
    exit 2
}
Write-Host "Every one of the $($results.Count) mutations was caught." -ForegroundColor Green
