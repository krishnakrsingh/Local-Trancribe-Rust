<#
.SYNOPSIS
    Measure word grouping on the real meeting, and report aggregates only.

.DESCRIPTION
    Runs the real production path (`vaani-cli transcribe`) over the real
    recording and reports what grouping did to it: the inter-word gap
    distribution, the segment count at every candidate pause threshold, and the
    before/after numbers.

    Two phases:

      1. WITH grouping, the way the product ships. This is the before/after
         measurement and the one that matters: it exercises the exact code path a
         user gets.
      2. With VAANI_SEGMENT_MEASURE set, which makes `vaani_segment` also emit
         the full gap distribution and a threshold sweep. That is where the pause
         default is chosen from.

    It prints counts, durations, distributions and timings. It never prints,
    logs or writes transcript text. The exports the CLI writes go to a directory
    under the system temp directory, never inside the repository, and this script
    counts their cues without reading their content.

    Runtime: about 90 s per phase on an RTX 3050 Ti, dominated by ASR.

.PARAMETER Clip
    The recording. Defaults to $VAANI_E2E_CLIP, then to scratch\clip.wav.

.PARAMETER Out
    Where the exports go. Defaults to a directory under the system temp
    directory. Never inside the repository.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\a11\measure_grouping.ps1
#>
[CmdletBinding()]
param([string]$Clip, [string]$Out)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

$CargoDir = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path -LiteralPath $CargoDir) { $env:Path = "$CargoDir;$env:Path" }

if (-not $Clip) { $Clip = $env:VAANI_E2E_CLIP }
if (-not $Clip) { $Clip = Join-Path $Root 'scratch\clip.wav' }
if (-not (Test-Path -LiteralPath $Clip -PathType Leaf)) {
    Write-Host "no recording at $Clip" -ForegroundColor Yellow
    exit 3
}

if (-not $Out) { $Out = Join-Path $env:TEMP 'vaani-a11-grouping' }
if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# `RuntimeLocation::discover` searches relative to the *executable*, walking up
# three and then four directories looking for `runtime/`. That works when the
# build output is inside the repository and fails for an external
# CARGO_TARGET_DIR, which is how the parallel agents build.
#
# The documented override does not help either: `VAANI_RUNTIME_DIR` is returned
# verbatim as the DLL *path* rather than having `onnxruntime.dll` joined onto
# it, so naming the directory makes the runtime report as absent. Naming the
# **DLL** is therefore the one form that works, and it is what this script uses.
# Both are defects in `vaani-inference`, which is frozen; see OPEN ISSUES.
$env:VAANI_ARTIFACTS_ROOT = Join-Path $Root 'artifacts'
$RtDll = Join-Path $Root 'runtime\onnxruntime.dll'
if (-not (Test-Path -LiteralPath $RtDll -PathType Leaf)) {
    Write-Host "no ONNX Runtime at $RtDll" -ForegroundColor Yellow
    exit 3
}
Remove-Item Env:VAANI_RUNTIME_DIR -ErrorAction SilentlyContinue
$env:VAANI_RUNTIME_DIR = $RtDll

# Strip ANSI colour so the fields can be parsed back out of the log.
function Get-Fields {
    param([string]$Path)
    $esc = [char]27
    $text = (Get-Content -LiteralPath $Path -Raw) -replace "$esc\[[0-9;]*[A-Za-z]", ''
    ($text -replace '[\r\n]+', ' ') -replace '\s+', ' '
}

Write-Host ''
Write-Host '=== VaaniDesk grouping measurement ===' -ForegroundColor Cyan
Write-Host "  clip  $Clip"
Write-Host "  out   $Out"
Write-Host ''

$failures = 0

# ---------------------------------------------------------------- phase 1
Write-Host '--- phase 1: the shipped path (grouping on) ---' -ForegroundColor Cyan
$Dir1 = Join-Path $Out 'shipped'
New-Item -ItemType Directory -Force -Path $Dir1 | Out-Null
$Log1 = Join-Path $Out 'shipped.log'
$env:VAANI_SEGMENT_MEASURE = $null
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& cargo run --release --quiet --bin vaani-cli -- --debug transcribe $Clip `
    --format all --out $Dir1 --overwrite *>&1 | Out-File -Encoding utf8 $Log1
$code = $LASTEXITCODE
$sw.Stop()
if ($code -ne 0) {
    Write-Host "FAILED: vaani-cli exited $code" -ForegroundColor Red
    exit 1
}
Write-Host ("  wall  {0:N1} s" -f $sw.Elapsed.TotalSeconds)

$flat = Get-Fields $Log1
$summary = [regex]::Match($flat, 'readable segments (.*?) job ')
if (-not $summary.Success) {
    Write-Host 'FAILED: grouping did not run.' -ForegroundColor Red
    exit 1
}
Write-Host ''
Write-Host '  what grouping did:'
foreach ($pair in ($summary.Groups[1].Value -split ' ')) {
    if ($pair -match '^([a-z_0-9]+)=(.*)$') { Write-Host ("    {0,-24} {1}" -f $Matches[1], $Matches[2]) }
}
foreach ($k in 'segments', 'speakers') {
    $m = [regex]::Match($flat, "(?m)^${k}: (.+)$")
    if ($m.Success) { Write-Host ("    {0,-24} {1}" -f $k, $m.Groups[1].Value) }
}

Write-Host ''
Write-Host '  exports (counted, never read):'
foreach ($f in Get-ChildItem -LiteralPath $Dir1 -File) {
    $cues = if ($f.Extension -in @('.srt', '.vtt')) {
        (Select-String -LiteralPath $f.FullName -Pattern ' --> ' -SimpleMatch).Count
    } else { '' }
    Write-Host ("    {0,-5} {1,9:N0} bytes {2}" -f $f.Extension.TrimStart('.'), $f.Length, $cues)
}

# ---------------------------------------------------------------- phase 2
Write-Host ''
Write-Host '--- phase 2: the tuning measurement (VAANI_SEGMENT_MEASURE=1) ---' -ForegroundColor Cyan
$Dir2 = Join-Path $Out 'tuning'
New-Item -ItemType Directory -Force -Path $Dir2 | Out-Null
$Log2 = Join-Path $Out 'tuning.log'
$env:VAANI_SEGMENT_MEASURE = '1'
$sw = [System.Diagnostics.Stopwatch]::StartNew()
# `--debug` is required: the CLI's log filter is `warn` without it, and the
# sweep is an `info` line.
& cargo run --release --quiet --bin vaani-cli -- --debug transcribe $Clip `
    --format srt --out $Dir2 --overwrite *>&1 | Out-File -Encoding utf8 $Log2
$code = $LASTEXITCODE
$sw.Stop()
$env:VAANI_SEGMENT_MEASURE = $null
if ($code -ne 0) {
    Write-Host "FAILED: vaani-cli exited $code" -ForegroundColor Red
    exit 1
}
Write-Host ("  wall  {0:N1} s" -f $sw.Elapsed.TotalSeconds)

$flat2 = Get-Fields $Log2
$gap = [regex]::Match($flat2, 'inter-word gap distribution (.*?) segmentation sweep')
if ($gap.Success) {
    Write-Host ''
    Write-Host '  inter-word gap distribution:'
    foreach ($pair in ($gap.Groups[1].Value -split ' ')) {
        if ($pair -match '^gap_([a-z_0-9]+)=(.*)$') { Write-Host ("    gap_{0,-22} {1}" -f $Matches[1], $Matches[2]) }
    }
    Write-Host ("    percentiles are p{0}" -f (($gap.Groups[1].Value -split ' ' | Where-Object { $_ -like 'gap_percentiles_ms=*' }) -replace 'gap_percentiles_ms=', ''))
}

$sweep = @($flat2 -split 'no diarization result\) ' | Select-Object -Skip 1)
if ($sweep.Count -gt 0) {
    Write-Host ''
    Write-Host '  pause-threshold sweep (pause rule alone, no diarization):'
    Write-Host ("    {0,-8} {1,-7} {2,-8} {3,-9} {4,-8} {5}" -f 'thr_ms', 'segs', 'w/seg', 'mean_ms', 'max_ms', 'breaks')
    foreach ($r in $sweep) {
        $t = if ($r -match 'threshold_ms=(\d+)') { $Matches[1] } else { continue }
        $s = if ($r -match 'segments=(\d+)') { $Matches[1] } else { '?' }
        $w = if ($r -match 'words_per_segment=([\d.]+)') { [math]::Round([double]$Matches[1], 2) } else { '?' }
        $mn = if ($r -match 'mean_duration_ms=([\d.]+)') { [math]::Round([double]$Matches[1]) } else { '?' }
        $mx = if ($r -match 'max_duration_ms=(\d+)') { $Matches[1] } else { '?' }
        $b = if ($r -match 'breaks=\[(.*?)\]') { $Matches[1] } else { '?' }
        Write-Host ("    {0,-8} {1,-7} {2,-8} {3,-9} {4,-8} {5}" -f $t, $s, $w, $mn, $mx, $b)
    }
} else {
    Write-Host '  no sweep in the log' -ForegroundColor Yellow
    $failures++
}

# ---------------------------------------------------------------- verdict
Write-Host ''
if ($failures -eq 0) {
    Write-Host 'MEASUREMENT OK' -ForegroundColor Green
    Write-Host "  logs in $Out"
    exit 0
}
Write-Host "MEASUREMENT INCOMPLETE ($failures checks failed)" -ForegroundColor Yellow
exit 1
