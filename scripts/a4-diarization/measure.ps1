# Run the vaani-diarization release benchmark and sample the *process* peak
# working set from outside, because the crate deliberately contains no `unsafe`
# and therefore cannot read its own working set.
#
# Usage:
#   powershell -File scripts/a4-diarization/measure.ps1 [-Clip]
#
# With -Clip it also measures the run over scratch/clip.wav by invoking the
# benchmark twice: once for the synthetic cases and once with the real
# recording, so the two peak figures are not conflated.

param(
    [switch]$Clip
)

$ErrorActionPreference = 'Stop'
$env:CARGO_TARGET_DIR = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { 'target' }

# The Rust toolchain is not on PATH in every shell this runs from, and a build
# that silently does nothing is worse than one that fails.
$cargo = if ($env:CARGO) { $env:CARGO } else { Join-Path $HOME '.cargo\bin\cargo.exe' }
if (-not (Test-Path -LiteralPath $cargo)) { throw "cargo not found at $cargo" }

# Build first, so the measured process is the benchmark and not a rustc that
# happens to be compiling it. A compile's peak working set is several hundred
# megabytes and would swamp the number we are after.
& $cargo build -p vaani-diarization --release --example benchmark
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$exe = Join-Path $env:CARGO_TARGET_DIR 'release\examples\benchmark.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "benchmark binary not found at $exe" }

$peak = 0
$proc = Start-Process -FilePath $exe -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\a4-bench.txt"
while (-not $proc.HasExited) {
    try {
        $proc.Refresh()
        if ($proc.WorkingSet64 -gt $peak) { $peak = $proc.WorkingSet64 }
    } catch {
        break
    }
    Start-Sleep -Milliseconds 20
}
$proc.WaitForExit()

Write-Output "--- benchmark output ---"
Get-Content "$env:TEMP\a4-bench.txt"
Write-Output "--- peak working set ---"
Write-Output ("{0:N1} MB" -f ($peak / 1MB))
Write-Output ("{0} bytes" -f $peak)

Remove-Item -LiteralPath "$env:TEMP\a4-bench.txt" -ErrorAction SilentlyContinue
