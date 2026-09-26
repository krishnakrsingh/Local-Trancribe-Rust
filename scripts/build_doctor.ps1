<#
.SYNOPSIS
    VaaniDesk build doctor: one command that says whether the tree is healthy.

.DESCRIPTION
    Runs the full gate in the cheapest-to-cheapest order and prints a concise
    report. Exists because "does this build work?" is asked constantly -- by CI,
    by a new agent, and by a human before starting a long task -- and answering
    it with four cargo invocations typed from memory is how people end up
    skipping the linter.

    Stages, in order:

      1. toolchain    rustc/cargo present, MSVC linkable, expected target
      2. format       cargo fmt --check
      3. check        cargo check --workspace --all-targets
      4. clippy       cargo clippy --workspace --all-targets  (warnings are errors)
      5. test         cargo test --workspace  (unit + integration, no model)
      6. runtime      ONNX Runtime present and loadable
      7. model        SraVaani artefacts present and consistent
      8. golden       front-end parity and end-to-end decode parity

    Stages 6-8 need the runtime and the model. Without them they report SKIP and
    say which command installs them; they never silently pass.

.PARAMETER Quick
    Stop after stage 5. Use in CI, where stages 6-8 have their own jobs.

.PARAMETER Release
    Use the release profile. Required for any stage whose timing is reported,
    because debug numbers are not release numbers.

.PARAMETER Json
    Emit a machine-readable report on stdout instead of the human one.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_doctor.ps1 -Quick -Json
#>
[CmdletBinding()]
param([switch]$Quick, [switch]$Release, [switch]$Json)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Cargo is installed to %USERPROFILE%\.cargo\bin by a `--no-modify-path`
# rustup install, which is the documented setup for this project.
$CargoDir = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path -LiteralPath $CargoDir) { $env:Path = "$CargoDir;$env:Path" }
$Cargo = 'cargo'
if (-not (Get-Command $Cargo -ErrorAction SilentlyContinue)) {
    Write-Host "cargo not found. Install Rust from https://rustup.rs" -ForegroundColor Red
    exit 2
}

$Profile = if ($Release) { '--release' } else { @() }
$Results = [System.Collections.Generic.List[object]]::new()

function Add-Result {
    param([string]$Stage, [string]$Status, [string]$Detail = '')
    $Results.Add([pscustomobject]@{ stage = $Stage; status = $Status; detail = $Detail })
}

function Invoke-Stage {
    param(
        [string]$Stage,
        [scriptblock]$Body,
        [switch]$Optional
    )
    if ($Optional) {
        $pre = $Body.Invoke()
        if ($null -eq $pre) {
            Add-Result $Stage 'SKIP' 'prerequisite not present'
            return
        }
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = & $Body 2>&1 | Out-String
    $sw.Stop()
    if ($LASTEXITCODE -eq 0) {
        Add-Result $Stage 'PASS' ("{0:N1}s" -f $sw.Elapsed.TotalSeconds)
    } else {
        Add-Result $Stage 'FAIL' 'see the tail below'
        # Print enough of the output to act on, not the whole thing: a failed
        # `cargo test` in a workspace with 110 tests is thousands of lines.
        $lines = $out -split "`r?`n" | Where-Object { $_ -match 'error|warning:|panicked|FAILED|assertion' }
        $lines | Select-Object -First 25 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
    }
}

# --- 1. toolchain ---------------------------------------------------------
$rustc = (& $Cargo --version 2>&1 | Out-String).Trim()
$target = (& $Cargo metadata --format-version 1 --no-deps 2>&1 | Out-String)
$toolchain = (Get-Content -LiteralPath (Join-Path $Root 'rust-toolchain.toml') -ErrorAction SilentlyContinue |
    Select-String -Pattern 'channel' | ForEach-Object { ($_ -split '"')[1] })
if ($rustc -match 'cargo (\S+)') {
    Add-Result 'toolchain' 'PASS' ("$($Matches[1]) (pinned $toolchain)")
} else {
    Add-Result 'toolchain' 'FAIL' $rustc
}

# A quick, real link test. `cargo check` never invokes the linker, so a missing
# MSVC or Windows SDK shows up as a confusing error much later -- usually on a
# CI machine, which is the worst place to find out.
$tmp = Join-Path $env:TEMP ("vaani-link-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
# Cargo requires src/main.rs, not main.rs at the root. Getting this wrong
# produces "no targets specified in the manifest", which reads like a cargo
# problem rather than a layout one.
New-Item -ItemType Directory -Force -Path (Join-Path $tmp 'src') | Out-Null
Set-Content -LiteralPath (Join-Path $tmp 'src\main.rs') -Value 'fn main() { println!("ok"); }'
Set-Content -LiteralPath (Join-Path $tmp 'Cargo.toml') -Value "[package]`nname=`"linktest`"`nversion=`"0.0.0`"`nedition=`"2021`""
$linkOut = (& $Cargo run --quiet --manifest-path (Join-Path $tmp 'Cargo.toml') 2>&1 | Out-String)
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
if ($linkOut -match 'ok') {
    Add-Result 'link' 'PASS' 'MSVC linker works'
} else {
    Add-Result 'link' 'FAIL' ($linkOut -split "`r?`n" | Select-Object -Last 3 | Out-String)
}

# --- 2..5 cargo gates -----------------------------------------------------
Invoke-Stage 'fmt' { & $Cargo fmt --all --check 2>&1 | Out-Null }
Invoke-Stage 'check' { & $Cargo check --workspace --all-targets @Profile 2>&1 | Out-Null }
Invoke-Stage 'clippy' { & $Cargo clippy --workspace --all-targets @Profile -- -D warnings 2>&1 | Out-Null }
Invoke-Stage 'test' { & $Cargo test --workspace @Profile 2>&1 | Out-Null }

if ($Quick) {
    if ($Json) { $Results | ConvertTo-Json -Depth 3 }
    else { $Results | Format-Table -AutoSize | Out-String | Write-Host }
    $bad = @($Results | Where-Object { $_.status -eq 'FAIL' }).Count
    exit ([Math]::Min($bad, 125))
}

# --- 6..8 runtime, model, golden ------------------------------------------
# Each of these is "optional" in the sense that a fresh checkout legitimately
# lacks it. None of them is optional in the sense of being allowed to pass
# silently: SKIP names the command that would satisfy it.
$runtimeTest = {
    & $Cargo test -p vaani-inference --test runtime_smoke @Profile -- --nocapture 2>&1 | Out-String
}
$out = & $runtimeTest
if ($out -match 'NOT FOUND' -or $out -match 'not found at') {
    Add-Result 'runtime' 'SKIP' 'run scripts\fetch_runtime.ps1'
} elseif ($out -match 'test result: FAILED') {
    Add-Result 'runtime' 'FAIL' 'the ONNX runtime smoke test failed'
} else {
    Add-Result 'runtime' 'PASS' 'ONNX Runtime present; see the smoke test output'
}

$art = Join-Path $Root 'artifacts\sravaani'
$haveModel = (Test-Path (Join-Path $art 'encoder.fp16.onnx')) -and
             (Test-Path (Join-Path $art 'decoder_joint.onnx')) -and
             (Test-Path (Join-Path $art 'preproc.json')) -and
             (Test-Path (Join-Path $art 'tokenizer.model'))
if (-not $haveModel) {
    Add-Result 'model' 'SKIP' 'run scripts\export_sravaani_onnx.py --src SraVaani-1.0 --out artifacts\sravaani'
    Add-Result 'golden' 'SKIP' 'needs the model'
} else {
    Add-Result 'model' 'PASS' (Join-Path $art 'artifacts present')

    $goldens = @(
        (Join-Path $Root 'tests\fixtures\sravaani\frontend_golden.json'),
        (Join-Path $Root 'tests\fixtures\sravaani\decode_golden.json')
    ) | Where-Object { Test-Path $_ }
    if ($goldens.Count -lt 2) {
        Add-Result 'golden' 'SKIP' 'run scripts\make_golden.py --src SraVaani-1.0 --out tests\fixtures --decode-audio <wav>'
    } else {
        $env:VAANI_GOLDEN_RAW = (Join-Path $Root 'scratch\golden_clip.f32')
        $goldenOut = (& $Cargo test -p vaani-asr @Profile --test frontend_parity --test sravaani_transcribe -- --nocapture --test-threads=1 2>&1 | Out-String)
        if ($goldenOut -match 'test result: FAILED') {
            Add-Result 'golden' 'FAIL' 'front-end or decode parity failed'
            ($goldenOut -split "`r?`n" | Where-Object { $_ -match 'panicked|differs|assertion|WER|EXACT' } |
                Select-Object -First 12) | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
        } else {
            $m = [regex]::Match($goldenOut, 'EXACT MATCH')
            $w = [regex]::Match($goldenOut, 'word timings: .*')
            $detail = if ($m.Success) { 'decode is an exact match' } else { 'parity tests passed' }
            if ($w.Success) { $detail += "; $($w.Value)" }
            Add-Result 'golden' 'PASS' $detail
        }
    }
}

# --- report ---------------------------------------------------------------
if ($Json) {
    $Results | ConvertTo-Json -Depth 3
} else {
    Write-Host ''
    Write-Host 'VaaniDesk build doctor' -ForegroundColor Cyan
    Write-Host ('-' * 78)
    foreach ($r in $Results) {
        $colour = switch ($r.status) {
            'PASS' { 'Green' }
            'SKIP' { 'DarkYellow' }
            default { 'Red' }
        }
        Write-Host ("{0,-12} {1,-5} {2}" -f $r.stage, $r.status, $r.detail) -ForegroundColor $colour
    }
    Write-Host ('-' * 78)
    $fails = @($Results | Where-Object { $_.status -eq 'FAIL' })
    $skips = @($Results | Where-Object { $_.status -eq 'SKIP' })
    $summary = switch ($fails.Count) {
        0 { "healthy ($($skips.Count) skipped)" }
        default { "$($fails.Count) FAILED" }
    }
    Write-Host $summary -ForegroundColor $(if ($fails.Count) { 'Red' } else { 'Green' })
}

exit ([Math]::Min(@($Results | Where-Object { $_.status -eq 'FAIL' }).Count, 125))
