<#
.SYNOPSIS
    The VaaniDesk end-to-end acceptance gate: the real clip, the real model.

.DESCRIPTION
    Every stage in this project had been measured on the real meeting recording
    individually. Nobody had run the whole chain on it. This is the one command
    that does, and it is the gate for the only question that matters: is the
    transcript that comes out the other end actually right?

    It runs, in order:

      1. the real production path   vaani-cli transcribe <clip> --format all
         (real ModelManager, real EngineRegistry, real CUDA provider), and
         checks the exports it wrote;
      2. the correctness gate        cargo test -p vaani-cli --test
         e2e_acceptance -- --ignored, which asserts monotonic word timings,
         correctly rebased chunks, consistent speaker attribution, an exact
         SQLite round trip, and valid SRT/VTT/JSON exports.

    Both runs use the same clip, and stage 1's reported segment count is
    compared against stage 2's, so the assertions are known to be describing
    the same output the CLI ships.

    Nothing here prints transcript text. The recording is a real private
    meeting; the gate reports counts, durations, timings and hashes only.

    EXIT CODES
      0  every check passed
      1  a check failed -- the transcript is not trustworthy
      2  usage error
      3  preflight failed: no recording, no runtime, or no weights, and the
         gate could not run at all. This is deliberately NOT 0: a machine
         without the 2.6 GB of weights must not be able to report success, but
         it also must not be a red herring, so the reason is printed and the
         exit code is distinct from a real failure.

.PARAMETER Clip
    The recording. Defaults to $VAANI_E2E_CLIP, then to scratch\clip.wav.

.PARAMETER Out
    Where the exports and the JSON report go. Defaults to $VAANI_E2E_OUT, then
    to a directory under the system temp directory. Never inside the repository:
    the exports contain the meeting, and the repository is a git checkout.

.PARAMETER SkipCliRun
    Run only the correctness gate. Useful in CI where the CLI run is covered
    elsewhere; the correctness gate is the part that must never be skipped.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\e2e\acceptance.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\e2e\acceptance.ps1 -Clip D:\media\standup.m4a
#>
[CmdletBinding()]
param(
    [string]$Clip,
    [string]$Out,
    [switch]$SkipCliRun
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

# --- locate the toolchain ---------------------------------------------------
$CargoDir = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path -LiteralPath $CargoDir) { $env:Path = "$CargoDir;$env:Path" }
$Cargo = 'cargo.exe'
if (-not (Get-Command $Cargo -ErrorAction SilentlyContinue)) {
    Write-Host 'cargo not found. Install Rust from https://rustup.rs' -ForegroundColor Red
    exit 2
}

function Preflight-Fail {
    param([string]$What, [string]$Why, [string]$How)
    Write-Host ''
    Write-Host "ACCEPTANCE NOT RUN: $What" -ForegroundColor Yellow
    Write-Host "  why:    $Why"
    Write-Host "  fix:    $How"
    Write-Host ''
    exit 3
}

# --- preflight --------------------------------------------------------------
if (-not $Clip) { $Clip = $env:VAANI_E2E_CLIP }
if (-not $Clip) { $Clip = Join-Path $Root 'scratch\clip.wav' }
if (-not (Test-Path -LiteralPath $Clip -PathType Leaf)) {
    Preflight-Fail 'no recording' "no media file at $Clip" 'pass -Clip <file>, or set VAANI_E2E_CLIP'
}
$Clip = (Resolve-Path -LiteralPath $Clip).Path

if (-not $Out) { $Out = $env:VAANI_E2E_OUT }
if (-not $Out) { $Out = Join-Path $env:TEMP 'vaani-e2e-acceptance' }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path -LiteralPath $Out).Path

$Encoder = Join-Path $Root 'artifacts\sravaani\encoder.fp16.onnx'
if (-not (Test-Path -LiteralPath $Encoder -PathType Leaf)) {
    Preflight-Fail 'no model weights' "no SraVaani artefacts under $Root\artifacts" `
        'see docs/agent/MODELS.md; ~887 MiB and a Hugging Face token are needed for a first run'
}

$RtDll = Join-Path $Root 'runtime\onnxruntime.dll'
if (-not (Test-Path -LiteralPath $RtDll -PathType Leaf)) {
    Preflight-Fail 'no ONNX Runtime' "no onnxruntime.dll under $Root\runtime" `
        'run scripts\fetch_runtime.ps1, or set VAANI_RUNTIME_DIR to a directory that has it'
}

# The artefacts root IS honoured as a root directory. The runtime directory is
# NOT: `vaani_inference::RuntimeLocation::discover` returns
# `VAANI_RUNTIME_DIR` verbatim as the *DLL path* instead of joining
# `onnxruntime.dll` onto it, so setting it makes the runtime look absent. That
# crate is frozen, so the variable is cleared here and the doctor check below
# is what actually decides. See OPEN ISSUES in the acceptance report.
$env:VAANI_ARTIFACTS_ROOT = Join-Path $Root 'artifacts'
Remove-Item Env:VAANI_RUNTIME_DIR -ErrorAction SilentlyContinue

Write-Host ''
Write-Host '=== VaaniDesk end-to-end acceptance ===' -ForegroundColor Cyan
Write-Host "  checkout  $Root"
Write-Host "  clip      $Clip ($([math]::Round((Get-Item $Clip).Length / 1MB, 1)) MiB)"
Write-Host "  out       $Out"
Write-Host ''

# `doctor` is the honest test of whether the runtime is reachable: it is the
# same code path the job will take, and it costs a third of a second. Checking
# for a file is not, because `discover` searches several places and a
# relocated CARGO_TARGET_DIR moves all of them.
$DoctorLog = Join-Path $Out 'doctor.log'
& $Cargo run --release --quiet --bin vaani-cli -- doctor --json 2>&1 |
    Tee-Object -FilePath $DoctorLog | Out-Null
$Doctor = Get-Content $DoctorLog -Raw | ConvertFrom-Json
if (-not $Doctor.runtime.runtime_present) {
    Preflight-Fail 'no ONNX Runtime' "onnxruntime.dll not reachable: $($Doctor.runtime.ort_dll)" `
        'run scripts\fetch_runtime.ps1. If runtime\ is present but unreachable, the build output ' +
        'directory is outside the repository: vaani-inference finds runtime\ by walking up from ' +
        'the executable, so the default target\ directory is required.'
}
Write-Host ("  runtime   {0}" -f $Doctor.summary)
Write-Host ("  machine   {0} logical CPUs, {1:N1} GiB RAM" -f `
    $Doctor.runtime.machine.logical_cpus, ($Doctor.runtime.machine.total_ram_bytes / 1GB))
Write-Host ''

$Failures = [System.Collections.Generic.List[string]]::new()

# --- stage 1: the real production path --------------------------------------
if (-not $SkipCliRun) {
    Write-Host '--- 1. the production CLI path, end to end ---' -ForegroundColor Cyan
    $CliDir = Join-Path $Out 'cli'
    if (Test-Path -LiteralPath $CliDir) { Remove-Item -LiteralPath $CliDir -Recurse -Force }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $CliLog = Join-Path $Out 'cli-transcribe.log'
    & $Cargo run --release --quiet --bin vaani-cli -- transcribe $Clip --format all --out $CliDir 2>&1 |
        Tee-Object -FilePath $CliLog
    $code = $LASTEXITCODE
    $sw.Stop()

    if ($code -ne 0) {
        $Failures.Add("vaani-cli transcribe exited $code; see $CliLog")
    }
    else {
        $SegmentLine = Select-String -Path $CliLog -Pattern '^segments: (\d+)$' |
            Select-Object -First 1
        $SpeakerLine = Select-String -Path $CliLog -Pattern '^speakers: (.+)$' |
            Select-Object -First 1
        $Exported = @(Get-ChildItem -LiteralPath $CliDir -File -ErrorAction SilentlyContinue)
        Write-Host ("  wall      {0:N1} s" -f $sw.Elapsed.TotalSeconds)
        Write-Host ("  segments  {0}" -f $(if ($SegmentLine) { $SegmentLine.Matches[0].Groups[1].Value } else { '?' }))
        Write-Host ("  speakers  {0}" -f $(if ($SpeakerLine) { $SpeakerLine.Matches[0].Groups[1].Value } else { '?' }))
        Write-Host ("  exported  {0} files" -f $Exported.Count)
        foreach ($f in $Exported) { Write-Host ("             {0,-5} {1,10:N0} bytes" -f $f.Extension.TrimStart('.'), $f.Length) }
        if ($Exported.Count -lt 5) {
            $Failures.Add("only $($Exported.Count) export files were written; five are required")
        }
        $env:VAANI_E2E_CLI_SEGMENTS = $(if ($SegmentLine) { $SegmentLine.Matches[0].Groups[1].Value } else { '' })
    }
    Write-Host ''
}

# --- stage 2: the correctness gate ------------------------------------------
Write-Host '--- 2. correctness: timings, rebasing, speakers, storage, exports ---' -ForegroundColor Cyan
$env:VAANI_E2E_CLIP = $Clip
$env:VAANI_E2E_OUT = $Out

& $Cargo test --release -p vaani-cli --test e2e_acceptance -- --ignored --nocapture 2>&1 |
    Tee-Object -FilePath (Join-Path $Out 'acceptance-test.log')
$testCode = $LASTEXITCODE

$Summary = Get-Content (Join-Path $Out 'acceptance-test.log') -Raw -ErrorAction SilentlyContinue
if ($Summary -match 'ACCEPTANCE SKIPPED') {
    Preflight-Fail 'the correctness gate skipped' 'see the reason printed above' 'fix what it names'
}
if ($testCode -ne 0) {
    $Failures.Add("the correctness gate failed; see acceptance-test.log")
}
if ($Summary -notmatch 'ACCEPTANCE OK') {
    $Failures.Add("the correctness gate did not reach ACCEPTANCE OK")
}

# The two runs must be describing the same output. If the CLI reports a
# different segment count than the gate measured, the gate is not testing the
# shipped path and the whole result is void.
if ($env:VAANI_E2E_CLI_SEGMENTS) {
    $Report = Join-Path $Out 'acceptance.json'
    if (Test-Path -LiteralPath $Report) {
        $J = Get-Content $Report -Raw | ConvertFrom-Json
        if ("$($J.segments)" -ne "$($env:VAANI_E2E_CLI_SEGMENTS)") {
            $Failures.Add(
                "the CLI reported $($env:VAANI_E2E_CLI_SEGMENTS) segments but the gate measured $($J.segments); " +
                'the gate is not exercising the shipped path')
        }
        else {
            Write-Host ''
            Write-Host "  both runs agree: $($J.segments) segments, $($J.words) words, $($J.speakers) speakers"
        }
    }
}
Write-Host ''

# --- verdict ----------------------------------------------------------------
if ($Failures.Count -gt 0) {
    Write-Host 'ACCEPTANCE FAILED' -ForegroundColor Red
    foreach ($f in $Failures) { Write-Host "  - $f" -ForegroundColor Red }
    Write-Host ''
    exit 1
}
Write-Host 'ACCEPTANCE PASSED' -ForegroundColor Green
Write-Host "  artifacts in $Out"
Write-Host ''
exit 0
