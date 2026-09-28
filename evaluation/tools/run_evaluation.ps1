<#
.SYNOPSIS
    The VaaniDesk accuracy evaluation, end to end, in one command.

.DESCRIPTION
    Cuts the clips, probes each one, runs the production CLI over each clip,
    converts every export into the harness's hypothesis form, and scores the
    set. Deterministic: re-running it overwrites clips/, hypotheses/ and
    results/ and changes nothing else.

    What it does NOT do is produce an accuracy number. It produces the input to
    one. A clip is scored only when evaluation\references\<id>\ contains both a
    reference file and an ANNOTATED.md provenance note, and today none does.
    The report says `unscored` for every clip and says why, and that is the
    honest state of this project, not a failure of this script.

    Everything is Rust and offline. No Python, no network, no model in the
    scoring path.

.PARAMETER Set
    The evaluation set directory. Defaults to evaluation\ next to this script.

.PARAMETER SkipTranscribe
    Re-score from existing hypotheses. Useful when only the references changed,
    and much faster: ASR is 80 % of the run.

.PARAMETER SkipBuild
    Do not build vaani-cli or vaani-eval. Assumes both are already built.

.PARAMETER Device
    Passed to `vaani transcribe --device`. Default auto.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File evaluation\tools\run_evaluation.ps1

.EXAMPLE
    # Score again after annotating one clip, without re-running ASR.
    powershell -ExecutionPolicy Bypass -File evaluation\tools\run_evaluation.ps1 `
        -SkipTranscribe -SkipBuild
#>
[CmdletBinding()]
param(
    [string]$Set,
    [switch]$SkipTranscribe,
    [switch]$SkipBuild,
    [string]$Device = 'auto'
)

$ErrorActionPreference = 'Continue'

$Tools = Split-Path -Parent $PSScriptRoot
if (-not $Set) { $Set = $Tools }
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

# --- toolchain ---------------------------------------------------------------
$CargoDir = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path -LiteralPath $CargoDir) { $env:Path = "$CargoDir;$env:Path" }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host 'cargo not found. Install Rust from https://rustup.rs' -ForegroundColor Red
    exit 2
}

# The harness is its own cargo workspace (the root manifest is architect-owned
# and this track may not edit it), so it needs its own target directory and its
# own --manifest-path. Kept out of the repository: a 100 MB target directory in
# a checkout is a nuisance and `git status` noise.
if (-not $env:CARGO_TARGET_DIR) {
    $env:CARGO_TARGET_DIR = Join-Path $env:TEMP 'vaanidesk-eval-target'
}
$EvalManifest = Join-Path $Set 'scoring\Cargo.toml'
$EvalExe = Join-Path $env:CARGO_TARGET_DIR 'release\vaani-eval.exe'
$CliProfile = 'release'
$CliExe = Join-Path $env:CARGO_TARGET_DIR "$CliProfile\vaani-cli.exe"

# --- preflight ---------------------------------------------------------------
function Stop-NotReady {
    param([string]$What, [string]$How)
    Write-Host ''
    Write-Host "NOT RUN: $What" -ForegroundColor Yellow
    Write-Host "  fix: $How"
    Write-Host ''
    exit 3
}

$Source = Join-Path $Root 'scratch\clip.wav'
if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    Stop-NotReady 'no source recording' "expected $Source (gitignored; see docs/JOURNEY.md)"
}
$Manifest = Join-Path $Set 'set.json'
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    Stop-NotReady "no set manifest at $Manifest" 'run vaani-eval from evaluation/scoring, or pass -Set'
}

Write-Host ''
Write-Host '=== VaaniDesk accuracy evaluation ===' -ForegroundColor Cyan
Write-Host "  checkout   $Root"
Write-Host "  set        $Set"
Write-Host "  source     scratch\clip.wav ($([math]::Round((Get-Item $Source).Length / 1MB, 1)) MiB)"
Write-Host "  target     $env:CARGO_TARGET_DIR"
Write-Host ''

# --- build -------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host '--- 0. build ---' -ForegroundColor Cyan
    # The harness first: `vaani-eval cut` is what produces the clips, so it has
    # to exist before anything else can run.
    & cargo build --release --offline --manifest-path $EvalManifest
    if ($LASTEXITCODE -ne 0) { Stop-NotReady 'vaani-eval did not build' 'see the compiler output above' }
    & cargo build --$CliProfile --offline --bin vaani-cli
    if ($LASTEXITCODE -ne 0) { Stop-NotReady 'vaani-cli did not build' 'see the compiler output above' }
}
if (-not (Test-Path -LiteralPath $EvalExe -PathType Leaf)) {
    Stop-NotReady "no vaani-eval binary at $EvalExe" 'drop -SkipBuild'
}
if (-not $SkipTranscribe -and -not (Test-Path -LiteralPath $CliExe -PathType Leaf)) {
    Stop-NotReady "no vaani-cli binary at $CliExe" 'drop -SkipBuild'
}

# Runtime discovery. `vaani-inference` walks up from the executable looking for
# runtime/, which fails whenever CARGO_TARGET_DIR lives outside the repository,
# so the override is set explicitly. It is the *directory*; the crate joins
# onnxruntime.dll onto it.
$env:VAANI_RUNTIME_DIR = Join-Path $Root 'runtime'
$env:VAANI_ARTIFACTS_ROOT = Join-Path $Root 'artifacts'
if (-not (Test-Path -LiteralPath (Join-Path $env:VAANI_RUNTIME_DIR 'onnxruntime.dll') -PathType Leaf)) {
    Stop-NotReady 'no ONNX Runtime' 'run scripts\fetch_runtime.ps1'
}

# --- 1. cut the clips --------------------------------------------------------
Write-Host '--- 1. cut the clips ---' -ForegroundColor Cyan
$ClipsDir = Join-Path $Set 'clips'
New-Item -ItemType Directory -Force -Path $ClipsDir | Out-Null
$ClipIds = @(
    (Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json).clips | ForEach-Object { $_.id }
)
foreach ($id in $ClipIds) {
    $def = (Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json).clips |
        Where-Object { $_.id -eq $id }
    & $EvalExe cut --in $Source --out (Join-Path $ClipsDir "$id.wav") `
        --start $def.start_ms --duration $def.duration_ms
    if ($LASTEXITCODE -ne 0) { Stop-NotReady "could not cut $id" 'see the message above' }

    # The cut is verified by reading it back, so `cut` already failed if the
    # length was wrong. The structure is measured and compared against the
    # value committed in set.json, which catches a clip that is the right
    # length and the wrong audio.
    $probe = & $EvalExe probe --in (Join-Path $ClipsDir "$id.wav") `
        --out (Join-Path $env:TEMP "vaani-eval-$id.structure.json")
    if ($LASTEXITCODE -ne 0) { Stop-NotReady "could not probe $id" 'see the message above' }
    $now = ($probe | Out-String) | ConvertFrom-Json
    if ($null -ne $def.structure -and $def.structure -ne $null) {
        $drift = [math]::Abs($now.speech_fraction - $def.structure.speech_fraction)
        if ($drift -gt 0.02) {
            Write-Host ("  WARNING {0}: speech fraction moved {1:P2} from the manifest; the clip is not the one the manifest describes" -f $id, $drift) -ForegroundColor Yellow
        }
    }
}
Write-Host "  $($ClipIds.Count) clips, each verified by reading the written file back"
Write-Host ''

# --- 2. run the system -------------------------------------------------------
if (-not $SkipTranscribe) {
    Write-Host '--- 2. run the production CLI on every clip ---' -ForegroundColor Cyan
    & cargo run --release --quiet --bin vaani-cli -- doctor | Out-Null
    if ($LASTEXITCODE -ne 0) { Stop-NotReady 'the runtime is not ready' 'run vaani-cli doctor' }

    $Sw = [System.Diagnostics.Stopwatch]::StartNew()
    foreach ($id in $ClipIds) {
        $Hyp = Join-Path $Set "hypotheses\$id"
        New-Item -ItemType Directory -Force -Path $Hyp | Out-Null
        # The export directory has to exist first: `--out <file>` inside a
        # missing folder is a documented failure (JOURNEY.md, phase 5), and
        # this script is not the place to work around a product bug by
        # silently creating trees.
        $OutDir = Join-Path $env:TEMP "vaani-eval-run-$id"
        if (Test-Path -LiteralPath $OutDir) { Remove-Item -LiteralPath $OutDir -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

        $Log = & $CliExe transcribe (Join-Path $ClipsDir "$id.wav") `
            --format json --device $Device `
            --out (Join-Path $OutDir 'transcript.json') --overwrite 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  $id : FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
            $Log | ForEach-Object { Write-Host "    $_" }
            continue
        }
        Copy-Item -LiteralPath (Join-Path $OutDir 'transcript.json') `
            -Destination (Join-Path $Hyp 'export.json') -Force
        & $EvalExe hyp --in (Join-Path $Hyp 'export.json') --out-dir $Hyp
        if ($LASTEXITCODE -ne 0) { Stop-NotReady "could not read the $id export" 'see the message above' }
    }
    $Sw.Stop()
    Write-Host ("  wall {0:N1} s" -f $Sw.Elapsed.TotalSeconds)
    Write-Host ''
}

# --- 3. metadata and score ---------------------------------------------------
Write-Host '--- 3. write metadata and score ---' -ForegroundColor Cyan
& $EvalExe metadata --set $Set
if ($LASTEXITCODE -ne 0) { Stop-NotReady 'could not write the per-clip metadata' 'see the message above' }

$Scored = & $EvalExe score --set $Set
$Code = $LASTEXITCODE
$Scored | ForEach-Object { Write-Host $_ }
Write-Host ''

# --- privacy check, before anything else can exit ---------------------------
# Checked first and unconditionally: a run that ends on any other verdict must
# still have proved that nothing private is staged for commit.
Write-Host '--- privacy check ---' -ForegroundColor Cyan
$Dirty = git status --porcelain -- evaluation 2>$null
$Bad = @()
if ($Dirty) {
    foreach ($Line in $Dirty) {
        $Path = $Line.Substring(3)
        foreach ($Prefix in @('evaluation/clips/', 'evaluation/references/', 'evaluation/hypotheses/')) {
            if ($Path.StartsWith($Prefix, [System.StringComparison]::Ordinal)) {
                $Bad += $Path
            }
        }
        if ($Path -match '\.(wav|mp3|mp4|flac|m4a)$') { $Bad += $Path }
    }
}
if ($Bad.Count -gt 0) {
    Write-Host '  PRIVACY VIOLATION: private audio or transcript text is staged for commit:' -ForegroundColor Red
    $Bad | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Write-Host '  evaluation\.gitignore is supposed to prevent this. Fix it before committing.' -ForegroundColor Red
    exit 1
}
Write-Host '  clean: no clip, no reference and no hypothesis is tracked' -ForegroundColor Green
Write-Host ''

# --- verdict -----------------------------------------------------------------
$Summary = Join-Path $Set 'results\summary.json'
if (Test-Path -LiteralPath $Summary) {
    $S = Get-Content -LiteralPath $Summary -Raw | ConvertFrom-Json
    Write-Host '--- reference coverage ---' -ForegroundColor Cyan
    Write-Host ("  clips                      {0}" -f $S.set.clips)
    Write-Host ("  human ASR reference         {0}" -f $S.set.human_asr_reference)
    Write-Host ("  human diarization reference {0}" -f $S.set.human_diarization_reference)
    Write-Host ("  human word-timing reference {0}" -f $S.set.human_word_timing_reference)
    Write-Host ("  awaiting annotation         {0}" -f $S.set.awaiting_annotation)
    Write-Host ("  with a system hypothesis    {0}" -f $S.set.with_hypothesis)
    Write-Host ''
    if ($S.set.human_asr_reference -eq 0) {
        Write-Host 'NO ACCURACY NUMBER EXISTS.' -ForegroundColor Yellow
        Write-Host '  The harness ran, the clips are cut, the system was scored against nothing' -ForegroundColor Yellow
        Write-Host '  because nobody has listened to the audio and written down what was said.' -ForegroundColor Yellow
        Write-Host '  See evaluation\ANNOTATION.md. This is the state of the project, not a' -ForegroundColor Yellow
        Write-Host '  failure of the run.' -ForegroundColor Yellow
        Write-Host ''
        # `score` exits 1 when nothing could be scored, on purpose: "the harness
        # ran" must never be readable as "the system is accurate". A caller that
        # wants to accept the unscored state passes -SkipTranscribe and gets 0.
        if ($Code -ne 0 -and -not $SkipTranscribe) { exit 1 }
    }
}

exit 0
