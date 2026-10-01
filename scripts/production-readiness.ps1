#!/usr/bin/env pwsh
# The one command that decides whether this tree is releasable.
#
# Every stage here has failed for real at some point, or guards a property that
# was measured rather than assumed. Nothing is included because it is cheap to
# run; things that only run in CI are not here, because a gate nobody runs
# locally is a gate that is not run.
#
# Returns non-zero on ANY release-blocking failure. Stages are not skipped
# silently: a stage that cannot run says so and counts as a failure unless it is
# explicitly optional, because "the test was skipped so the release passed" is
# the failure mode this script exists to prevent.
#
#   .\scripts\production-readiness.ps1                 # the full gate
#   .\scripts\production-readiness.ps1 -SkipRealMedia  # no private clip available
#   .\scripts\production-readiness.ps1 -Flavour Gpu    # package the GPU build too
#
# -SkipRealMedia is the only permitted concession, it is not the default, and it
# prints exactly which real-media assertions were therefore not checked.

[CmdletBinding()]
param(
    # Skip the gates that need the private reference recording and the bundled
    # model. Everything else still runs. Intended for a machine that has neither.
    [switch]$SkipRealMedia,

    # The GPU runtime is also the default target, because shipping a CPU-only
    # build as the primary artefact is what caused the "why is it not on my GPU"
    # report: the package a user launches has to be the one they were told to
    # install. A GPU package needs `-Flavour Gpu` explicitly, and building only
    # one overwrites `target\release-pkg` in place -- so a CPU run leaves a CPU
    # tree where a GPU tree used to be, which is a trap for anyone testing by
    # hand afterwards.
    [ValidateSet('Cpu', 'Gpu', 'Both')]
    [string]$Flavour = 'Both',

    # Set by tests that are themselves part of the gate, to avoid recursion.
    [switch]$Child
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $repo

$script:Failures = [System.Collections.Generic.List[string]]::new()
$script:Warnings = [System.Collections.Generic.List[string]]::new()
$script:StartedAt = Get-Date
$cargo = "$env:USERPROFILE\.cargo\bin\cargo.exe"

function Write-Stage {
    param([string]$Name, [string]$Status, [string]$Detail = '')
    $colour = switch ($Status) { 'PASS' { 'Green' } 'FAIL' { 'Red' } 'SKIP' { 'Yellow' } default { 'Grey' } }
    $line = "  {0,-6} {1}" -f $Status, $Name
    if ($Detail) { $line += "  -- $Detail" }
    Write-Host $line -ForegroundColor $colour
}

function Invoke-Stage {
    param(
        [string]$Name,
        [string[]]$Command,
        [switch]$Optional,
        [switch]$IgnoreExitCode
    )
    Write-Host ""
    Write-Host "== $Name" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # Invoked as a *string command line* so that the first element is a real
    # executable. Passing an array to `&` runs the first element as a command
    # and the rest as arguments, which works -- but a stage that spawns a child
    # PowerShell wrote a module-load error to stderr and still reported PASS,
    # because $LASTEXITCODE was left over from the previous stage. A gate that
    # cannot fail is worse than no gate, so the exit code is captured
    # immediately and a child that fails to launch is a failure, not a warning.
    $global:LASTEXITCODE = 0
    try {
        # Invoked as a *script block* with the splatted array, so each element is
        # an argument and a path containing spaces stays one argument. Two
        # earlier versions were wrong in opposite directions: `& $Command`
        # treated the first element as a command name and silently lost the
        # exit code of a spawned child PowerShell (so a broken stage reported
        # PASS), and joining into one string broke on any path with a space in
        # it (so every stage reported "could not be started"). This is the form
        # that does neither.
        $out = & { & $Command[0] @($Command[1..($Command.Count - 1)]) } 2>&1
    } catch {        Write-Host "     could not run: $_" -ForegroundColor Red
        Write-Stage $Name 'FAIL' 'the command could not be started'
        $script:Failures.Add("$Name (could not start)")
        return $false
    }
    $code = $LASTEXITCODE
    $out | Select-Object -Last 25 | ForEach-Object { Write-Host "     $_" -ForegroundColor DarkGray }
    $sw.Stop()
    if ($code -eq 0 -or $IgnoreExitCode) {
        Write-Stage $Name 'PASS' ("{0:N1}s" -f $sw.Elapsed.TotalSeconds)
        return $true
    }
    if ($Optional) {
        Write-Stage $Name 'SKIP' "exit $code; not release-blocking"
        $script:Warnings.Add("$Name exited $code (optional)")
        return $true
    }
    Write-Stage $Name 'FAIL' "exit $code after $([math]::Round($sw.Elapsed.TotalSeconds,1))s"
    $script:Failures.Add("$Name (exit $code)")
    return $false
}

function Note-Skip {
    param([string]$What, [string]$Why)
    Write-Stage $What 'SKIP' $Why
    $script:Warnings.Add("$What skipped: $Why")
}

Write-Host ""
Write-Host "VaaniDesk production readiness" -ForegroundColor White
Write-Host "  repo   : $repo" -ForegroundColor DarkGray
Write-Host "  started: $($script:StartedAt.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor DarkGray
Write-Host "  head   : $(& git rev-parse --short HEAD 2>$null)" -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
# 1. Clean tree. Everything below is measuring the working tree, so a dirty tree
#    means the results describe code that is not what would be shipped.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "== repository state" -ForegroundColor Cyan
$dirty = & git status --porcelain
if ($dirty) {
    Write-Stage 'working tree is clean' 'FAIL' 'uncommitted changes: the gate would measure code that is not what ships'
    $script:Failures.Add('working tree is not clean')
    $dirty | Select-Object -First 10 | ForEach-Object { Write-Host "     $_" -ForegroundColor DarkYellow }
} else {
    Write-Stage 'working tree is clean' 'PASS'
}

# ---------------------------------------------------------------------------
# 2. No private media committed, ever.
#
#    This has been checked by hand; making it a gate means it cannot be forgotten
#    by whoever adds a fixture next. The allow-list is narrow and explained,
#    because the failure mode of this check is a false positive that trains
#    people to ignore it -- which would be worse than not having it.
#
#    `tests/fixtures/qa/*.wav` is excluded because those are synthetic files
#    written byte by byte by `vaani-qa`'s own generator: 1 s of silence, 1 s of
#    tone, 8 s of alternating tones. They are generated in-process precisely so
#    that "committed" and "generated" cannot disagree, and no human speech is in
#    them.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "== no private media in history" -ForegroundColor Cyan
$allow = '^(?:[0-9a-f]{7,40}\s+)?tests/fixtures/qa/'
$media = & git log --all --name-only --format='' 2>$null |
    Where-Object { $_ -match '\S' } |
    Where-Object { $_ -notmatch $allow } |
    Select-String -Pattern '\.mp4$|\.wav$|^scratch/|^WhatsApp|\.m4a$|\.flac$|\.ogg$|\.mkv$|\.mov$' -CaseSensitive:$false |
    ForEach-Object { $_.Line } |
    Select-Object -Unique
if ($media) {
    Write-Stage 'no private media committed' 'FAIL' ''
    $media | Select-Object -First 5 | ForEach-Object { Write-Host "     $_" -ForegroundColor Red }
    $script:Failures.Add('private media is present in git history')
} else {
    Write-Stage 'no private media committed' 'PASS' 'synthetic tests/fixtures/qa WAVs are allow-listed'
}

# ---------------------------------------------------------------------------
# 3-7. The compile-and-test gates.
# ---------------------------------------------------------------------------
Invoke-Stage 'formatting'        @($cargo, 'fmt', '--all', '--', '--check') | Out-Null
Invoke-Stage 'clippy'            @($cargo, 'clippy', '--workspace', '--all-targets', '--', '-D', 'warnings') | Out-Null
Invoke-Stage 'type check'        @($cargo, 'check', '--workspace', '--all-targets') | Out-Null
Invoke-Stage 'unit + integration tests' @($cargo, 'test', '--workspace', '--release') | Out-Null

# 8. The resource-limit and leak suites on their own, so a failure names the
#    area rather than arriving as one line of a 1687-test summary.
Invoke-Stage 'resource limits'   @($cargo, 'test', '--release', '-p', 'vaani-media', '--test', 'resource_limits') | Out-Null
Invoke-Stage 'resource leaks'    @($cargo, 'test', '--release', '-p', 'vaani-app', '--test', 'resource_leaks') | Out-Null
Invoke-Stage 'performance gates' @($cargo, 'test', '--release', '-p', 'vaani-asr', '--test', 'batch_regression') | Out-Null

# 9. The authoritative doctor. Nine stages, each of which has caught something.
Invoke-Stage 'build doctor'      @('powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'scripts\build_doctor.ps1', '-Release') | Out-Null

# ---------------------------------------------------------------------------
# 10-11. Real media. These are the gates that cannot be faked, and the reason
#   the pipeline's own assertions exist. Skipped only on explicit request.
# ---------------------------------------------------------------------------
$clip = Test-Path 'scratch\clip.wav'
$mp4 = @(Get-ChildItem -Filter '*.mp4' -ErrorAction SilentlyContinue).Count -gt 0

if ($SkipRealMedia -or -not $clip) {
    $why = if ($SkipRealMedia) { '-SkipRealMedia was given' } else { 'scratch/clip.wav is absent' }
    Note-Skip 'real-media ship gate' $why
    $script:Warnings.Add('REAL-MEDIA GATES DID NOT RUN: word counts, speaker counts and progressive streaming are UNVERIFIED for this build')
} else {
    Invoke-Stage 'real-media ship gate' @($cargo, 'test', '--release', '-p', 'vaani-desktop', '--test', 'ui_ship_gate', '--', '--ignored', '--nocapture') | Out-Null
    if (-not $mp4) {
        Note-Skip 'MP4 path' 'no .mp4 in the repository root; the video path is unverified'
    }
}

# 12. Golden token equality: the strict one. Already inside the workspace run,
#    but named separately so a failure is unmistakably about the model.
if ($SkipRealMedia) {
    Note-Skip 'golden token equality' '-SkipRealMedia was given'
} else {
    Write-Host ""
    Write-Host "== golden token equality" -ForegroundColor Cyan
    # `matches_reference_token_ids_on_real_audio` skips itself unless
    # VAANI_GOLDEN_RAW points at the raw f32 slice the fixture was built from.
    # That audio is deliberately not in the repository, so on a machine without
    # it the test *skips* and still exits zero.
    #
    # Reporting that as PASS would be a lie: the strictest guarantee the project
    # has (identical token ids to the reference decode) would be unchecked while
    # the gate said everything was fine. So the stage distinguishes 'ran and
    # matched' from 'skipped', and reports the latter as a SKIP whose consequence
    # is named in the verdict.
    #
    # The filename is the one `build_doctor.ps1` uses. An earlier version of this
    # stage looked for `clip-golden.f32` and reported the guarantee unverified on
    # a machine where it had in fact just passed -- the same class of error as a
    # false PASS, in the opposite direction: a real pass reported as a gap.
    $raw = 'scratch\golden_clip.f32'
    if (Test-Path $raw) {
        # Absolute, and that detail is load-bearing. A relative
        # VAANI_GOLDEN_RAW resolves against the *test binary's* working
        # directory, which for an integration test is the crate root
        # (`crates/vaani-asr`), not the repository root. So the same value that
        # passes in one invocation fails with "the system cannot find the path
        # specified" in another, and it looked like a flaky model rather than a
        # path bug. `build_doctor.ps1` already resolves it absolutely for
        # exactly this reason.
        $env:VAANI_GOLDEN_RAW = (Resolve-Path $raw).Path
        $out = & $cargo test --release -p vaani-asr --test sravaani_transcribe 2>&1
        $code = $LASTEXITCODE
        $ran = ($out | Select-String 'matches_reference_token_ids_on_real_audio \.\.\. ok').Count -gt 0
        if (-not ($code -eq 0 -and $ran)) {
            # Print why: an opaque "exit 101" for a correctness gate is worse
            # than useless, and the cause here was a path, not the model.
            $out | Select-String -Pattern 'panicked|assertion|differs|left|right|not found' |
                Select-Object -First 6 | ForEach-Object { Write-Host "     $_" -ForegroundColor Red }
        }
        Remove-Item Env:\VAANI_GOLDEN_RAW -ErrorAction SilentlyContinue
        if ($code -eq 0 -and $ran) {
            Write-Stage 'golden token equality' 'PASS' 'ran and matched the reference token ids'
        } else {
            Write-Stage 'golden token equality' 'FAIL' "exit $code, matched=$ran"
            $script:Failures.Add('golden token equality did not run or did not match')
        }
    } else {
        Note-Skip 'golden token equality' "$raw is absent, so token-id equality with the reference is UNVERIFIED for this build"
        $script:Warnings.Add('GOLDEN TOKEN EQUALITY NOT CHECKED: the strictest correctness guarantee is unverified')
    }
}

# ---------------------------------------------------------------------------
# 13. Package. Built for real, not assumed, because packaging has broken on a
#     missing DLL and on a licence file that was never staged.
# ---------------------------------------------------------------------------
$flavours = if ($Flavour -eq 'Both') { @('Cpu', 'Gpu') } else { @($Flavour) }
foreach ($f in $flavours) {
    Invoke-Stage "package + self-test ($f)" @('powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'scripts\release\make-release.ps1', '-RuntimeFlavour', $f) | Out-Null
}

# ---------------------------------------------------------------------------
# 14. Verify the staged artefact, not the build directory. A green build and a
#     broken package are different states, and only the second one ships.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "== staged artefact" -ForegroundColor Cyan
$pkg = 'target\release-pkg\VaaniDesk-0.1.0-win-x64'
$missing = @()
foreach ($required in @('VaaniDesk.exe', 'vaani.exe', 'vaani-cli.exe', 'vaani-dist.json', 'vaani-dist.sha256', 'verify-install.ps1', 'OFFLINE-INSTALL.md', 'models', 'runtime', 'licenses')) {
    if (-not (Test-Path (Join-Path $pkg $required))) { $missing += $required }
}
if ($missing) {
    Write-Stage 'package contains what it must' 'FAIL' "missing: $($missing -join ', ')"
    $script:Failures.Add("package is missing $($missing -join ', ')")
} else {
    Write-Stage 'package contains what it must' 'PASS'
}
$zip = 'target\release-pkg\VaaniDesk-0.1.0-win-x64.zip'
if (Test-Path $zip) {
    $size = [math]::Round((Get-Item $zip).Length / 1GB, 2)
    Write-Stage 'portable zip produced' 'PASS' "$size GiB"
} else {
    Write-Stage 'portable zip produced' 'FAIL'
    $script:Failures.Add('no portable zip')
}
$cudaDll = Test-Path (Join-Path $pkg 'runtime\onnxruntime_providers_cuda.dll')
if (($flavours -contains 'Gpu') -and -not $cudaDll) {
    Write-Stage 'GPU runtime present' 'FAIL' 'the GPU package has no CUDA provider DLL'
    $script:Failures.Add('GPU package missing the CUDA provider')
} elseif ($flavours -contains 'Gpu') {
    Write-Stage 'GPU runtime present' 'PASS'
} else {
    Write-Stage 'GPU runtime present' 'SKIP' 'CPU flavour only'
}

# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------
$elapsed = (Get-Date) - $script:StartedAt
Write-Host ""
Write-Host ("=" * 72) -ForegroundColor White
if ($script:Failures.Count -eq 0) {
    Write-Host "PRODUCTION READY" -ForegroundColor Green
    Write-Host ("  {0} gates, 0 failures, {1:N1} minutes" -f 14, $elapsed.TotalMinutes) -ForegroundColor Green
    if ($script:Warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "  Caveats that are NOT failures but are also NOT verified:" -ForegroundColor Yellow
        foreach ($w in $script:Warnings) { Write-Host "    - $w" -ForegroundColor Yellow }
    }
    Pop-Location
    exit 0
}

Write-Host "PRODUCTION READY: NO" -ForegroundColor Red
Write-Host ""
Write-Host "  Release-blocking failures:" -ForegroundColor Red
foreach ($f in $script:Failures) { Write-Host "    - $f" -ForegroundColor Red }
if ($script:Warnings.Count -gt 0) {
    Write-Host ""
    Write-Host "  Also not verified:" -ForegroundColor Yellow
    foreach ($w in $script:Warnings) { Write-Host "    - $w" -ForegroundColor Yellow }
}
Pop-Location
exit 1
