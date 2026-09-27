<#
.SYNOPSIS
    Runs A6's real-size benchmark: store, load, search and export a transcript
    with the shape a real 13-minute meeting produces.

.DESCRIPTION
    The benchmark lives in `crates/vaani-export/tests/real_size.rs` as an
    ignored test, because it writes a ~50 MB database and takes about ten
    seconds. This script runs it in release mode, which is the only mode whose
    numbers mean anything for a performance question.

    It reads `scratch/clip.wav` (the private meeting) read-only, to time the one
    storage operation that touches the media file: hashing it for the
    duplicate-transcription check. No transcript content is produced, printed or
    stored by it, and the database it builds is deleted when the test ends.

.PARAMETER Debug
    Run the benchmark unoptimised. Slower, and the numbers are not comparable
    with the reported ones; useful only for checking that it still runs.

.EXAMPLE
    pwsh scripts/a6/bench.ps1
#>
[CmdletBinding()]
param(
    [switch]$Debug
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $repo
try {
    # A private target directory, so eight agents benchmarking at once do not
    # serialise on one cargo build lock.
    $env:CARGO_TARGET_DIR = if ($env:VAANI_BENCH_TARGET) {
        $env:VAANI_BENCH_TARGET
    } else {
        Join-Path ([System.IO.Path]::GetTempPath()) 'vaani-a6-bench-target'
    }

    $profile = if ($Debug) { '' } else { '--release' }
    Write-Host "target dir : $env:CARGO_TARGET_DIR"
    Write-Host "profile    : $(if ($Debug) { 'debug' } else { 'release' })"
    Write-Host "scratch    : $(if (Test-Path (Join-Path $repo 'scratch/clip.wav')) { 'clip.wav present (read-only)' } else { 'clip.wav absent' })"
    Write-Host ''

    & cargo test -p vaani-export --test real_size @profile -- --ignored --nocapture
    if ($LASTEXITCODE -ne 0) {
        throw "the benchmark failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
