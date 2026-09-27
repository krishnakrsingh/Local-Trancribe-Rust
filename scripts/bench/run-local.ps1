<#
.SYNOPSIS
    Run the A9 staged benchmark in the release profile.

.DESCRIPTION
    Release-profile runner for `vaani-bench` (backing `vaani benchmark <file>
    [--json]`). A debug RTF is not a performance claim about anything: the
    workspace sets opt-level 3, thin LTO and codegen-units 1, and a debug
    build of the engine is roughly fifty times slower.

    PRIVATE FIXTURE PATH CONVENTION (read carefully):
    - The 12-minute private meeting recording travels ONLY as the -File
      argument (or $env:VAANI_BENCH_FILE, which this script resolves to that
      argument). It is NEVER committed: *.mp4/*.wav and scratch/ are
      gitignored, and this script refuses to run `git add` on anything.
    - Default convention when -File is omitted: $env:VAANI_BENCH_FILE, else
      scratch\bench-meeting.* (first match). That path is local-only by
      construction (.gitignore covers scratch/ plus every media extension).
    - This script only stats the file. Nothing is uploaded, transcribed, or
      copied anywhere.

    Measurements are printed by vaani-bench itself (human table, or one JSON
    line with -Json). Any number published from that output must state the
    machine and the release profile (docs/agent/PERFORMANCE.md).

.PARAMETER File
    Path to the private meeting recording. Local only, never committed.

.PARAMETER Json
    Print one machine-readable BenchReport JSON line to stdout instead of the
    human table.

.PARAMETER SkipBuild
    Skip the `cargo build --release` step and run the existing binary. Useful
    when re-running against the same tree.

.EXAMPLE
    $env:VAANI_BENCH_FILE = "D:\private\meeting-12min.mp4"
    powershell -ExecutionPolicy Bypass -File scripts\bench\run-local.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bench\run-local.ps1 -File "D:\private\meeting-12min.mp4" -Json
#>
[CmdletBinding()]
param(
    [string] $File = "",
    [switch] $Json,
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'

# Toolchain bootstrap (docs/agent/PERFORMANCE.md repro steps): a fresh
# `powershell -File` invocation may not have cargo on PATH.
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
    if (Test-Path -LiteralPath (Join-Path $cargoBin 'cargo.exe')) {
        $env:Path += ";$cargoBin"
    }
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error 'cargo not found. Install the pinned toolchain (rust-toolchain.toml) and retry.'
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')
$manifest = Join-Path $repoRoot 'crates\vaani-bench\Cargo.toml'
if (-not (Test-Path -LiteralPath $manifest)) {
    Write-Error "vaani-bench manifest not found at $manifest."
}

# The private fixture travels only as an argument (or env var resolved here to
# an argument). Never committed, never defaulted to a repo-tracked path.
if ([string]::IsNullOrWhiteSpace($File)) {
    $File = $env:VAANI_BENCH_FILE
}
if ([string]::IsNullOrWhiteSpace($File)) {
    $candidate = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'scratch') -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'bench-meeting.*' } |
        Select-Object -First 1)
    if ($candidate.Count -gt 0) {
        $File = $candidate[0].FullName
    }
}
if ([string]::IsNullOrWhiteSpace($File)) {
    Write-Error ('No input file. Pass -File <path> or set $env:VAANI_BENCH_FILE. ' +
        'The 12-minute private meeting recording stays local: it is passed as a path argument only and is never committed.')
}
if (-not (Test-Path -LiteralPath $File)) {
    Write-Error "Input file not found: $File"
}
$resolved = (Resolve-Path -LiteralPath $File).Path
Write-Host "vaani-bench input: $resolved"

# Guardrail: refuse to run against a path that looks committed (tracked by
# git), so a fixture can never silently become a repo file. Paths outside the
# repo cannot be tracked, so they skip the git call entirely.
$repoPath = $repoRoot.Path
$isUnderRepo = ($resolved -eq $repoPath) -or
    $resolved.StartsWith($repoPath + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)
if ($isUnderRepo) {
    # `git ls-files --error-unmatch` exits nonzero for untracked paths (and
    # writes to stderr); that is the expected case, not a failure. Scope
    # $ErrorActionPreference so the stderr record stays non-terminating under
    # this script's global 'Stop' setting.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $null = & git -C $repoPath ls-files --error-unmatch -- "$resolved" 2>$null
    $trackedExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($trackedExit -eq 0) {
        Write-Error ("Refusing to benchmark a git-tracked file ($resolved). The private fixture must stay " +
            'untracked and gitignored; pass a local-only path instead.')
    }
}

# A9-owned target dir: eight agents must not serialise on one cargo lock.
$env:CARGO_TARGET_DIR = Join-Path $repoRoot 'target\a9'

if (-not $SkipBuild) {
    Write-Host 'Building vaani-bench (release)...'
    & cargo build --release --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { Write-Error 'cargo build --release -p vaani-bench failed.' }
}

$bin = Join-Path $env:CARGO_TARGET_DIR 'release\vaani-bench.exe'
if (-not (Test-Path -LiteralPath $bin)) {
    Write-Error "Expected binary not found at $bin (build step skipped or failed)."
}

$binArgs = @($resolved)
if ($Json) { $binArgs += '--json' }
Write-Host "Running: $bin $($binArgs -join ' ')"
& $bin @binArgs
exit $LASTEXITCODE
