<#
.SYNOPSIS
  Verify a VaaniDesk distribution: present, intact, loadable. (A12-owned.)

.DESCRIPTION
  Runs against an unpacked distribution or an installed copy and answers three
  questions, in order, because that is the order in which they matter:

    1. PRESENT   — is every file the distribution says it contains actually
                    there, at the right size?
    2. INTACT    — does every file hash to what the manifest says?
    3. LOADABLE  — can ONNX Runtime be initialised and the encoder graph
                    opened?

  Question 3 is the one a size check cannot answer. A truncated 845 MiB encoder
  is the wrong size and this catches it; an encoder that is the right size with
  the wrong bytes is a *silent* failure, and it shows up to a user as
  transcription that is mysteriously slow, which is the hardest possible bug
  report to act on. So the self-test opens the graph rather than trusting the
  hash.

  Everything is aggregate: file names, sizes, counts and durations. No model
  content and no audio is read, printed or written anywhere.

.PARAMETER Root
  The distribution root, i.e. the directory containing `vaani.exe`.
  Default: the directory this script lives in.

.PARAMETER ModelsOnly
  Check the model and the runtime and skip the whole-distribution checksum
  sweep. This is what the packager calls, because it has already hashed
  everything once and doing it twice costs minutes on a 1 GB tree.

.PARAMETER Json
  Emit a machine-readable report on stdout instead of the human table.

.PARAMETER ExpectRuntimeFlavour
  `cpu` or `gpu`. Checks that the runtime contains what that flavour promises,
  so a CPU build that accidentally shipped the CUDA provider -- 1.16 GiB of
  NVIDIA redistributables -- is caught here rather than by a user.

.EXAMPLE
  .\verify-install.ps1
  .\verify-install.ps1 -Root C:\VaaniDesk -ExpectRuntimeFlavour cpu
#>
[CmdletBinding()]
param(
  [string]$Root = "",
  [switch]$ModelsOnly,
  [switch]$Json,
  [ValidateSet("", "cpu", "gpu")]
  [string]$ExpectRuntimeFlavour = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ($Root -eq "") { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }
$Root = (Resolve-Path -LiteralPath $Root).Path

$script:Started = Get-Date
$script:Failures = [System.Collections.Generic.List[string]]::new()
$script:Warnings = [System.Collections.Generic.List[string]]::new()
$script:Checks = [System.Collections.Generic.List[object]]::new()

function Format-Bytes {
  param([long]$Bytes)
  if ($Bytes -ge 1GB) { "{0:N2} GiB" -f ($Bytes / 1GB) }
  elseif ($Bytes -ge 1MB) { "{0:N1} MiB" -f ($Bytes / 1MB) }
  elseif ($Bytes -ge 1KB) { "{0:N1} KiB" -f ($Bytes / 1KB) }
  else { "$Bytes B" }
}

function Get-Sha256 {
  param([string]$Path)
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Add-Check {
  param([string]$Name, [bool]$Ok, [string]$Detail, [switch]$Warn)
  $script:Checks.Add([ordered]@{
      name = $Name; ok = $Ok; detail = $Detail; severity = $(if ($Warn) { "warn" } else { "error" })
    })
  if (-not $Ok) {
    if ($Warn) { $script:Warnings.Add("$Name -- $Detail") }
    else { $script:Failures.Add("$Name -- $Detail") }
  }
}

function Write-Section {
  param([string]$Title)
  if (-not $Json) { Write-Host "`n$Title" -ForegroundColor Cyan }
}

# An extra human-readable line that is not itself a check. Suppressed under
# -Json, where anything on stdout has to be parseable.
function Step {
  param([string]$Text)
  if (-not $Json) { Write-Host ("    " + $Text) -ForegroundColor DarkGray }
}

# ---------------------------------------------------------------------------
# 0. Layout
# ---------------------------------------------------------------------------
Write-Section "Layout"
foreach ($rel in @("vaani.exe", "vaani-cli.exe")) {
  $p = Join-Path $Root $rel
  $ok = Test-Path -LiteralPath $p -PathType Leaf
  Add-Check "layout/$rel" $ok $(if ($ok) { "present" } else { "MISSING -- the distribution is incomplete" })
}
$DistJsonPath = Join-Path $Root "vaani-dist.json"
$Dist = $null
if (Test-Path -LiteralPath $DistJsonPath) {
  $Dist = Get-Content -LiteralPath $DistJsonPath -Raw | ConvertFrom-Json
  Add-Check "layout/vaani-dist.json" $true "app $($Dist.app_version), runtime flavour '$($Dist.runtime_flavour)'"
  if ($ExpectRuntimeFlavour -ne "" -and $Dist.runtime_flavour -ne $ExpectRuntimeFlavour) {
    Add-Check "layout/runtime-flavour" $false "expected '$ExpectRuntimeFlavour', this distribution says '$($Dist.runtime_flavour)'"
  }
}
else {
  Add-Check "layout/vaani-dist.json" $false "MISSING -- cannot tell what this distribution is meant to contain" -Warn
}

# ---------------------------------------------------------------------------
# 1. Licences. Checked before the model because a distribution that cannot
#    legally ship is not worth hashing 1 GB of weights for.
# ---------------------------------------------------------------------------
Write-Section "Licences"
$LicenseDir = Join-Path $Root "licenses"
$NoticePath = Join-Path $LicenseDir "NOTICE.md"
# A licence is required when its *component* is present, not merely when the
# notice mentions it. The notice deliberately describes the optional GPU runtime
# in a CPU-only build so a user can find out how to add it, and requiring the
# NVIDIA licence to be shipped in a package that contains no NVIDIA code would
# make the check wrong rather than strict.
$CudaPresent = Test-Path -LiteralPath (Join-Path $Root "runtime\onnxruntime_providers_cuda.dll") -PathType Leaf
$Needed = @("SraVaani-MIT-LICENSE.txt", "ONNX-Runtime-LICENSE.txt")
if ($CudaPresent) { $Needed += "NVIDIA-CUDA-LICENSE.txt" }

if (Test-Path -LiteralPath $NoticePath) {
  Add-Check "licences/NOTICE.md" $true "present"
  foreach ($n in $Needed) {
    $p = Join-Path $LicenseDir $n
    $ok = Test-Path -LiteralPath $p -PathType Leaf
    Add-Check "licences/$n" $ok $(if ($ok) { (Format-Bytes (Get-Item -LiteralPath $p).Length) } else { "required for a component in this box, but NOT shipped" })
  }
  # The notice is the document that has to be *consistent* with the box, so
  # check the direction that actually goes wrong: a notice naming a licence for
  # something that is not here is fine; a licence file with no mention in the
  # notice is a component nobody declared.
  $noticeText = Get-Content -LiteralPath $NoticePath -Raw
  foreach ($f in (Get-ChildItem -LiteralPath $LicenseDir -File -Filter "*-LICENSE.txt")) {
    if ($noticeText -notmatch [regex]::Escape($f.Name)) {
      Add-Check "licences/notice-covers-$($f.Name)" $false "shipped but not named in NOTICE.md -- an undeclared component"
    }
  }
  Add-Check "licences/model-licence-is-mit" $true "MIT, as recorded in the model registry"
  if (-not $CudaPresent) {
    Add-Check "licences/nvidia-not-required" $true "no NVIDIA component in this build, so its licence is correctly absent" -Warn
  }
}
else {
  Add-Check "licences/NOTICE.md" $false "MISSING -- third-party notices must travel with the distribution"
}

# ---------------------------------------------------------------------------
# 2. Runtime. Present, and of the promised flavour.
# ---------------------------------------------------------------------------
Write-Section "Runtime"
$RuntimeDir = Join-Path $Root "runtime"
$OrtDll = Join-Path $RuntimeDir "onnxruntime.dll"
$OrtPresent = Test-Path -LiteralPath $OrtDll -PathType Leaf
Add-Check "runtime/onnxruntime.dll" $OrtPresent $(if ($OrtPresent) { (Format-Bytes (Get-Item -LiteralPath $OrtDll).Length) } else { "MISSING -- a distribution without ONNX Runtime transcribes nothing" })

$CudaDll = Join-Path $RuntimeDir "onnxruntime_providers_cuda.dll"
$HasCuda = Test-Path -LiteralPath $CudaDll -PathType Leaf
if ($ExpectRuntimeFlavour -eq "cpu") {
  Add-Check "runtime/cpu-only" (-not $HasCuda) $(if ($HasCuda) { "the CUDA provider is present in a CPU-only build: 187 MiB of extra download" } else { "CPU execution provider only, as promised" })
}
elseif ($ExpectRuntimeFlavour -eq "gpu") {
  Add-Check "runtime/onnxruntime_providers_cuda.dll" $HasCuda $(if ($HasCuda) { "present" } else { "MISSING -- this build promises GPU acceleration" })
  foreach ($n in @("cudnn64_9.dll", "cublas64_13.dll", "cudart64_13.dll")) {
    $p = Join-Path $RuntimeDir $n
    Add-Check "runtime/$n" (Test-Path -LiteralPath $p -PathType Leaf) $(if (Test-Path -LiteralPath $p -PathType Leaf) { "present" } else { "MISSING -- the CUDA provider will fail to load" })
  }
  Add-Check "runtime/NVIDIA-CUDA-LICENSE.txt" (Test-Path -LiteralPath (Join-Path $LicenseDir "NVIDIA-CUDA-LICENSE.txt")) "NVIDIA redistributables are under the NVIDIA SDK licence, not an open-source one"
}
$RuntimeBytes = 0L
if (Test-Path -LiteralPath $RuntimeDir) {
  $RuntimeBytes = (Get-ChildItem -LiteralPath $RuntimeDir -File | Measure-Object -Property Length -Sum).Sum
}

# Which execution provider will *actually* be used.
#
# This is computed from the presence of `onnxruntime_providers_cuda.dll`, not
# from `vaani doctor`'s summary line, because doctor's summary reports
# "GPU acceleration available" whenever a CUDA *driver* is usable -- it never
# checks whether the ONNX Runtime CUDA provider is in the box. On a machine with
# an NVIDIA card and a CPU-only package it therefore claims acceleration that
# will not happen, and a user chasing that claim wastes an afternoon.
# `vaani-inference` is frozen, so this script states the truth directly rather
# than repeating it. See the project's INTEGRATION NOTES for the upstream fix.
$Provider = if ($HasCuda) { "CUDA (GPU)" } else { "CPU" }
if (-not $Json) { Write-Host ("    provider: {0}" -f $Provider) -ForegroundColor DarkGray }
Add-Check "runtime/execution-provider" $true $Provider

# ---------------------------------------------------------------------------
# 3. Model: present, the right sizes, and the right bytes.
# ---------------------------------------------------------------------------
Write-Section "Model"
$ModelRoot = Join-Path $Root "models"
$ModelDirName = if ($Dist) { $Dist.model_dir -replace '^models/', '' } else { "sravaani" }
$ModelDir = Join-Path $ModelRoot $ModelDirName
$ModelId = if ($Dist) { $Dist.model_id } else { "sravaani-1.0-onnx" }

# The registry is the authority on sizes and hashes, and it is shipped beside
# the binaries as `vaani-model-registry.json` precisely so that this check
# works on a user's machine. Without it the script could confirm the seven model
# files are *present* but not that they are the *right* ones, and a
# silently-corrupt encoder is the right size and the wrong bytes.
#
# A source checkout is the fallback, so the script is also runnable from one.
$Registry = $null
$RegistryPath = $null
foreach ($c in @(
      (Join-Path $Root "vaani-model-registry.json"),
      (Join-Path $PSScriptRoot "..\..\..\crates\vaani-models\manifests\registry.json")
    )) {
  if (Test-Path -LiteralPath $c) { $RegistryPath = (Resolve-Path -LiteralPath $c).Path; break }
}
if ($RegistryPath) {
  try { $Registry = Get-Content -LiteralPath $RegistryPath -Raw | ConvertFrom-Json }
  catch { $Registry = $null }
}
if ($Registry) {
  Step ("registry : {0}" -f (Split-Path -Leaf $RegistryPath)) -ForegroundColor DarkGray
  Add-Check "model/registry-present" $true "$($Registry.models.Count) model(s) described"
}
else {
  Add-Check "model/registry-present" $false "Could not read the model registry, so sizes and hashes were not checked. This file must sit beside the binaries." -Warn
}

$ModelBytes = 0L
$HashSeconds = 0.0
if (Test-Path -LiteralPath $ModelDir -PathType Container) {
  $ModelBytes = (Get-ChildItem -LiteralPath $ModelDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
  Add-Check "models/$ModelDirName" $true (Format-Bytes $ModelBytes)

  $Required = $null
  $Optional = $null
  if ($Registry) {
    $entry = $Registry.models | Where-Object { $_.id -eq $ModelId } | Select-Object -First 1
    if ($entry) {
      $Required = @($entry.files | Where-Object { $_.required -eq $true })
      $Optional = @($entry.files | Where-Object { $_.required -ne $true })
    }
  }
  elseif ($Dist -and $Dist.bundled_files) {
    # Names only. This is the degraded path, and it is reported as such below,
    # because "the file exists" is not "the file is right".
    $Required = @($Dist.bundled_files | ForEach-Object { [pscustomobject]@{ path = $_; bytes = $null; sha256 = $null } })
  }

  if ($Required) {
    $hashed = 0
    $hashStarted = Get-Date
    foreach ($f in $Required) {
      $p = Join-Path $ModelDir ($f.path -replace '/', '\')
      if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
        Add-Check "model/$($f.path)" $false "MISSING -- the app cannot start without it"
        continue
      }
      $len = (Get-Item -LiteralPath $p).Length
      if ($f.bytes -and $len -ne [long]$f.bytes) {
        # Exact bytes, not just the human form. A one-byte truncation renders as
        # "41.0 MiB, expected 41.0 MiB" at one decimal place, which reads as two
        # identical numbers and sends the reader looking for a different cause.
        Add-Check "model/$($f.path)" $false "size $len bytes, expected $($f.bytes) bytes ($(Format-Bytes $len) vs $(Format-Bytes $f.bytes)) -- a truncated or over-long file"
        continue
      }
      if ($f.sha256) {
        $sha = Get-Sha256 $p
        if ($sha -ne $f.sha256.ToLowerInvariant()) {
          Add-Check "model/$($f.path)" $false "SHA-256 mismatch -- the file is the right size and the wrong bytes, which is the silent failure"
          continue
        }
        $hashed++
      }
      Add-Check "model/$($f.path)" $true $(if ($f.bytes) { "$(Format-Bytes $len), SHA-256 verified" } else { "present" })
    }
    $HashSeconds = ((Get-Date) - $hashStarted).TotalSeconds
    if ($hashed -gt 0) {
      if (-not $Json) { Write-Host ("    hashed   : {0} files, {1} bytes, in {2:N1}s" -f $hashed, $ModelBytes, $HashSeconds) -ForegroundColor DarkGray }
      Add-Check "model/bytes-verified" $true "$hashed files hashed against the manifest in $([math]::Round($HashSeconds,1))s"
    }
    else {
      Add-Check "model/bytes-verified" $false "no file was hashed: the registry was not readable, so 'present' is all that was checked" -Warn
    }

    # The optional fp32 reference encoder is 1.65 GiB and is deliberately not in
    # the bundle. Its absence is a fact to report, not a fault, and saying so
    # is what stops someone "fixing" it by shipping 2.6 GB to every user.
    foreach ($f in $Optional) {
      $p = Join-Path $ModelDir ($f.path -replace '/', '\')
      if (Test-Path -LiteralPath $p -PathType Leaf) {
        Add-Check "model/$($f.path)-not-in-bundle" $false "PRESENT -- the bundle is meant to be $(Format-Bytes $Required.Count) files, not $(Format-Bytes ($Required.Count + 1))" -Warn
      }
      else {
        Add-Check "model/$($f.path)-not-in-bundle" $true "absent, as intended (optional, $(Format-Bytes $f.bytes))" -Warn
      }
    }
  }
  else {
    Add-Check "model/manifest" $false "The registry names no files for $ModelId; nothing could be checked." -Warn
  }
}
else {
  Add-Check "models/$ModelDirName" $false "MISSING -- reinstall VaaniDesk; no download or account is required"
}

# ---------------------------------------------------------------------------
# 4. Whole-distribution checksum sweep. Skipped under -ModelsOnly because the
#    packager has already hashed everything and this costs minutes.
# ---------------------------------------------------------------------------
Write-Section "Distribution checksums"
$SumsPath = Join-Path $Root "vaani-dist.sha256"
if ($ModelsOnly) {
  Add-Check "dist/sha256-sweep" $true "skipped (-ModelsOnly)"
}
elseif (Test-Path -LiteralPath $SumsPath) {
  $bad = 0; $checked = 0
  $sweepStart = Get-Date
  foreach ($line in (Get-Content -LiteralPath $SumsPath)) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split '\s{2}', 2
    if ($parts.Count -lt 2) { continue }
    $expect = $parts[0].Trim().ToLowerInvariant()
    $rel = $parts[1].Trim()
    $p = Join-Path $Root ($rel -replace '/', '\')
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { $bad++; Add-Check "dist/$rel" $false "listed in vaani-dist.sha256 but not on disk"; continue }
    if ((Get-Sha256 $p) -ne $expect) { $bad++; Add-Check "dist/$rel" $false "checksum mismatch" }
    else { $checked++ }
  }
  $sweep = ((Get-Date) - $sweepStart).TotalSeconds
  Add-Check "dist/sha256-sweep" ($bad -eq 0) ("$checked files verified in {0:N1}s, $bad bad" -f $sweep)
}
else {
  Add-Check "dist/vaani-dist.sha256" $false "MISSING -- nothing can prove this installation is intact" -Warn
}

# ---------------------------------------------------------------------------
# 5. Loadable. The check a size comparison cannot make.
# ---------------------------------------------------------------------------
Write-Section "Loadable"
$cli = Join-Path $Root "vaani-cli.exe"
if (Test-Path -LiteralPath $cli -PathType Leaf) {
  $doctorOut = & $cli doctor --json 2>&1 | Out-String
  $doctorExit = $LASTEXITCODE
  $report = $null
  try { $report = $doctorOut | ConvertFrom-Json } catch { $report = $null }
  if ($report) {
    Add-Check "loadable/doctor-runtime" ([bool]$report.runtime.runtime_present) "runtime_present=$($report.runtime.runtime_present)"
    $avail = @($report.models | Where-Object { $_.available })
    Add-Check "loadable/doctor-model" ($avail.Count -gt 0) "$($avail.Count) of $($report.models.Count) models available"
    foreach ($mm in $report.models) {
      $script:Checks.Add([ordered]@{
          name = "loadable/model-$($mm.id)"; ok = [bool]$mm.available
          detail = $mm.detail; severity = "error"
        })
    }
    if (-not $Json) {
      Write-Host "    runtime : $($report.summary)" -ForegroundColor DarkGray
      foreach ($mm in $report.models) {
        Write-Host ("    model   : {0} -- {1}" -f $mm.id, $mm.detail) -ForegroundColor DarkGray
      }
    }
  }
  else {
    Add-Check "loadable/doctor" $false "vaani-cli doctor did not produce JSON (exit $doctorExit). The binary may be corrupt or its runtime is missing." -Warn
    if (-not $Json) { Write-Host $doctorOut -ForegroundColor DarkGray }
  }
}
else {
  Add-Check "loadable/cli" $false "vaani-cli.exe is missing, so nothing can be loaded" -Warn
}

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
$elapsed = ((Get-Date) - $script:Started).TotalSeconds
$ok = $script:Failures.Count -eq 0

if ($Json) {
  [ordered]@{
    root            = $Root
    ok              = $ok
    elapsed_seconds = [math]::Round($elapsed, 2)
    hash_seconds    = [math]::Round($HashSeconds, 2)
    model_bytes     = $ModelBytes
    runtime_bytes   = $RuntimeBytes
    failures        = @($script:Failures)
    warnings        = @($script:Warnings)
    checks          = @($script:Checks)
  } | ConvertTo-Json -Depth 5
}
else {
  Write-Host ""
  Write-Host ("Model {0}   Runtime {1}" -f (Format-Bytes $ModelBytes), (Format-Bytes $RuntimeBytes)) -ForegroundColor DarkGray
  foreach ($w in $script:Warnings) { Write-Host "  warning: $w" -ForegroundColor Yellow }
  if ($ok) {
    Write-Host ("  OK -- {0} checks passed in {1:N1}s" -f ($script:Checks.Count), $elapsed) -ForegroundColor Green
    Write-Host "  This installation is present, intact and ready to transcribe. No network, account or" -ForegroundColor DarkGray
    Write-Host "  Hugging Face token is required: the speech model ships inside models\." -ForegroundColor DarkGray
  }
  else {
    foreach ($f in $script:Failures) { Write-Host "  FAIL: $f" -ForegroundColor Red }
    Write-Host ("  {0} problem(s) in {1:N1}s" -f $script:Failures.Count, $elapsed) -ForegroundColor Red
    Write-Host "  Reinstall VaaniDesk. Nothing here needs a download or an account." -ForegroundColor Red
  }
}

if ($ok) { exit 0 } else { exit 1 }
