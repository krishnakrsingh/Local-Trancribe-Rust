<#
.SYNOPSIS
  VaaniDesk Windows x64 release packager (A12-owned: scripts/release/**).

.DESCRIPTION
  Builds a distribution that works on a clean machine with **no network, no
  model directory and no Python**, and then proves it before it is written.

  1. `cargo build --release -p vaani-cli --bins`
  2. Read `crates/vaani-models/manifests/registry.json` and stage exactly the
     files it marks `required`, under `models/<dir>/`.
  3. Hash every staged weight against the manifest. **A mismatch, a missing
     file, or an absent `artifacts/` tree is a hard failure** unless
     `-AllowMissingWeights` is passed, and that switch also suppresses the
     installer so a weights-less build can never be shipped by accident.
  4. Stage the runtime, CPU-only by default (`-RuntimeFlavour Cpu`) or with the
     CUDA/cuDNN redistributables (`Gpu`).
  5. Write `vaani-dist.sha256` â€” a checksum manifest over **every file in the
     distribution** â€” plus `vaani-dist.json` describing the layout.
  6. Stage `licenses/`, `OFFLINE-INSTALL.md` and `verify-install.ps1`.
  7. Compress the portable ZIP, build the Inno Setup installer when ISCC.exe is
     available, and write `SHA256SUMS` over the top-level artefacts.

  Step 3 is the part that matters and it is deliberately strict. A8's version
  copied whatever it found and warned when it found nothing, so a release built
  on a machine without the weights produced a plausible-looking installer that
  could not transcribe. The failure mode this replaces is a user paying for a
  download and then being told, on a machine with no network, that the model
  was missing.

.PARAMETER Version
  Release version. Defaults to the vaani-core package version.

.PARAMETER OutDir
  Where artefacts land. Default: <CARGO_TARGET_DIR>\release-pkg.

.PARAMETER ModelSource
  Directory holding the SraVaani export. Default: <repo>\artifacts\sravaani.
  This is a *build input* and is never committed.

.PARAMETER RuntimeSource
  Directory holding the ONNX Runtime and CUDA redistributables fetched by the
  frozen scripts\fetch_runtime.ps1. Default: <repo>\runtime.

.PARAMETER RuntimeFlavour
  `Cpu` (default) or `Gpu`.
    Cpu -> runtime\onnxruntime.dll + onnxruntime_providers_shared.dll only.
    Gpu -> every DLL in RuntimeSource, adding the CUDA provider, cuBLAS, cuDNN
           and nvrtc.
  The default is Cpu because the CUDA set is 1.16 GiB of NVIDIA redistributables
  that only users with an NVIDIA GPU can use, and because NVIDIA's SDK licence
  section 2.5 is not compatible with shipping them inside an MIT-licensed
  no-charge product. See installer\licenses\NOTICE.md.

.PARAMETER SkipBuild
  Stage whatever is already built (faster re-runs).

.PARAMETER SkipInstaller
  Do not invoke ISCC.exe even when present.

.PARAMETER AllowMissingWeights
  Stage binaries and runtime without the model weights, for a local smoke test.
  Refuses to build an installer, refuses to write a `.sha256` manifest, and
  says loudly in its output that the result is not shippable.

.PARAMETER IncludeOptional
  Also stage the manifest's optional files. Off by default, and the default is
  the product decision: SraVaani's only optional file is the 1.65 GiB fp32
  reference encoder that no runtime path opens, and including it would make the
  distribution 2.55 GiB instead of 903 MiB.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\release\make-release.ps1
#>
[CmdletBinding()]
param(
  [string]$Version = "",
  [string]$OutDir = "",
  [string]$ModelSource = "",
  [string]$RuntimeSource = "",
  [ValidateSet("Cpu", "Gpu")]
  [string]$RuntimeFlavour = "Cpu",
  [switch]$SkipBuild,
  [switch]$SkipInstaller,
  [switch]$AllowMissingWeights,
  [switch]$IncludeOptional
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if ($Version -eq "") {
  $cargoToml = Get-Content (Join-Path $RepoRoot "crates\vaani-core\Cargo.toml") -Raw
  if ($cargoToml -match '(?m)^version\s*=\s*"([^"]+)"') { $Version = $Matches[1] }
  else { throw "Could not read a version from crates/vaani-core/Cargo.toml; pass -Version explicitly." }
}

# The release output goes under the target directory, never under `target\` in
# the repo: the worktree and the main checkout share a repository, and dropping
# a 1 GB installer into it makes `git status` unreadable for everyone else.
$DefaultOut = if ($env:CARGO_TARGET_DIR) { Join-Path $env:CARGO_TARGET_DIR "release-pkg" }
else { Join-Path $RepoRoot "target\release-pkg" }
if ($OutDir -eq "") { $OutDir = $DefaultOut }

if ($ModelSource -eq "") { $ModelSource = Join-Path $RepoRoot "artifacts\sravaani" }
if ($RuntimeSource -eq "") { $RuntimeSource = Join-Path $RepoRoot "runtime" }

$TargetDir = if ($env:CARGO_TARGET_DIR) { Join-Path $env:CARGO_TARGET_DIR "release" }
else { Join-Path $RepoRoot "target\release" }

$StageName = "VaaniDesk-$Version-win-x64"
$StageDir = Join-Path $OutDir $StageName
$ModelStage = Join-Path $StageDir "models"
$RuntimeStage = Join-Path $StageDir "runtime"
$LicenseStage = Join-Path $StageDir "licenses"

# The files that make the CPU execution provider work. ONNX Runtime's core DLL
# contains the CPU EP unconditionally; the CUDA EP is a separate DLL loaded only
# when it is present, so these two are the whole of a working CPU runtime.
$CpuRuntimeDlls = @("onnxruntime.dll", "onnxruntime_providers_shared.dll")

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

# Write UTF-8 with no byte-order mark.
#
# `-Encoding utf8NoBOM` is PowerShell 6+; on 5.1 -- which is the PowerShell
# shipped in Windows 11 and the only one a user or a Windows CI runner will
# have -- the value does not resolve and `Set-Content` throws before writing.
# A8's recovered version of this script used it in five places, so the script
# had never been run end to end on its target platform. This writes the same
# bytes on both.
#
# The BOM matters, not just for tidiness: `vaani-dist.json` is parsed by
# `verify-install.ps1` and potentially by a third party, and a leading U+FEFF
# makes `ConvertFrom-Json` fail on a strict reader. A checksum manifest whose
# first hash is preceded by three invisible bytes is a checksum manifest that
# fails to verify.
function Write-Utf8NoBom {
  param([string]$Path, [string[]]$Lines)
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllLines($Path, [string[]]$Lines, $enc)
}

# ---------------------------------------------------------------------------
# 1. Read the manifest. The file list is never written out by hand in this
#    script: a hand-copied list is a second source of truth that survives the
#    day someone adds a file to the manifest and forgets the packager, and the
#    failure is an installer that is missing a weights file.
# ---------------------------------------------------------------------------
$RegistryPath = Join-Path $RepoRoot "crates\vaani-models\manifests\registry.json"
if (-not (Test-Path -LiteralPath $RegistryPath)) {
  throw "Model registry not found at $RegistryPath. The packager refuses to guess the file list."
}
$Registry = Get-Content -LiteralPath $RegistryPath -Raw | ConvertFrom-Json
if ($Registry.models.Count -ne 1) {
  throw "Expected exactly one model in the registry, found $($Registry.models.Count). The bundle layout is defined for one bundled model; extend this script before adding another."
}
$Model = $Registry.models[0]
$Required = @($Model.files | Where-Object { $_.required -eq $true })
$Optional = @($Model.files | Where-Object { $_.required -ne $true })
$WeightsBytes = ($Required | Measure-Object -Property bytes -Sum).Sum

Write-Host "VaaniDesk release $Version"
Write-Host "  repo:        $RepoRoot"
Write-Host "  out:         $OutDir"
Write-Host "  stage:       $StageDir"
Write-Host ("  model:       {0} -- {1} required files, {2}, {3}" -f $Model.id, $Required.Count, (Format-Bytes $WeightsBytes), $Model.license.spdx)
Write-Host "  runtime:     $RuntimeFlavour"

# ---------------------------------------------------------------------------
# Locate cargo rather than assuming it is on PATH.
#
# A packaging script that only works when the developer remembered to open a
# Rust shell is a packaging script that works on CI and nowhere else, and the
# failure is a confusing "cargo is not recognized" halfway through a build. CI
# puts it on PATH; a plain Windows terminal on a machine with rustup installed
# does not, so the last two fallbacks are what make this runnable as written.
# ---------------------------------------------------------------------------
function Resolve-Cargo {
  if ($env:CARGO -and (Test-Path -LiteralPath $env:CARGO)) { return $env:CARGO }
  $cmd = Get-Command cargo.exe -ErrorAction SilentlyContinue
  if ($null -ne $cmd) { return $cmd.Source }
  $up = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
  if (Test-Path -LiteralPath $up) { return $up }
  throw "cargo not found. Install Rust from https://rustup.rs, or pass -SkipBuild if the binaries are already built."
}
$Cargo = Resolve-Cargo

# ---------------------------------------------------------------------------
# 2. Build.
# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
  Write-Host "`n[1/7] Building" -ForegroundColor Cyan
  Push-Location $RepoRoot
  try {
    & $Cargo build --release -p vaani-cli --bins
    if ($LASTEXITCODE -ne 0) { throw "cargo build --release failed." }
  }
  finally { Pop-Location }
}
else { Write-Host "`n[1/7] Building ... skipped (-SkipBuild)" }

$vaaniExe = Join-Path $TargetDir "vaani.exe"
$cliExe = Join-Path $TargetDir "vaani-cli.exe"
foreach ($bin in @($vaaniExe, $cliExe)) {
  if (-not (Test-Path -LiteralPath $bin)) { throw "Missing build output: $bin (run without -SkipBuild)." }
}

# ---------------------------------------------------------------------------
# 3. Stage the tree.
# ---------------------------------------------------------------------------
Write-Host "`n[2/7] Staging" -ForegroundColor Cyan
if (Test-Path -LiteralPath $StageDir) { Remove-Item -LiteralPath $StageDir -Recurse -Force }
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null
New-Item -ItemType Directory -Path $ModelStage -Force | Out-Null
New-Item -ItemType Directory -Path $RuntimeStage -Force | Out-Null
New-Item -ItemType Directory -Path $LicenseStage -Force | Out-Null

Copy-Item -LiteralPath $vaaniExe -Destination (Join-Path $StageDir "vaani.exe")
Copy-Item -LiteralPath $cliExe -Destination (Join-Path $StageDir "vaani-cli.exe")

# --- model weights, verified against the manifest -------------------------
$ModelDir = Join-Path $ModelStage $Model.dir
New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null

$Missing = [System.Collections.Generic.List[string]]::new()
$BadHash = [System.Collections.Generic.List[string]]::new()
$Staged = [System.Collections.Generic.List[string]]::new()

foreach ($f in $Required) {
  $src = Join-Path $ModelSource ($f.path -replace '/', '\')
  $dst = Join-Path $ModelDir ($f.path -replace '/', '\')
  if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
    $Missing.Add($f.path)
    continue
  }
  $len = (Get-Item -LiteralPath $src).Length
  if ($len -ne [long]$f.bytes) {
    # A size mismatch means the exporter produced something different from what
    # the manifest describes. Hashing it anyway would only produce a second,
    # more confusing error.
    $BadHash.Add("$($f.path): size $len, manifest says $($f.bytes)")
    continue
  }
  New-Item -ItemType Directory -Path (Split-Path -Parent $dst) -Force | Out-Null
  $sha = Get-Sha256 $src
  if ($sha -ne $f.sha256.ToLowerInvariant()) {
    $BadHash.Add("$($f.path): sha256 $sha, manifest says $($f.sha256)")
    continue
  }
  Copy-Item -LiteralPath $src -Destination $dst
  $Staged.Add($f.path)
}

foreach ($f in $Optional) {
  # Optional files are NOT staged by default.
  #
  # SraVaani's optional `encoder.onnx` is the 1.65 GiB fp32 reference encoder:
  # the app never opens it, it exists so the fp16 graph can be compared against
  # its own reference. Shipping it would make the distribution 2.55 GiB instead
  # of 903 MiB -- nearly three times the download for 0 % of the product. The
  # developer tree happens to have it because the exporter wrote it there; that
  # is a fact about the build machine, not about the product.
  #
  # An earlier version of this script staged optional files whenever the source
  # tree had them, and produced a 2.55 GiB "bundle" while reporting 886.9 MiB of
  # required weights. `-IncludeOptional` is the explicit opt-in for a developer
  # who wants a parity-checkable portable copy and is willing to carry the size.
  $src = Join-Path $ModelSource ($f.path -replace '/', '\')
  $have = Test-Path -LiteralPath $src -PathType Leaf
  if ($IncludeOptional -and $have) {
    $dst = Join-Path $ModelDir ($f.path -replace '/', '\')
    New-Item -ItemType Directory -Path (Split-Path -Parent $dst) -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst
    Write-Host ("  staged (optional, -IncludeOptional): {0}  {1}" -f $f.path, (Format-Bytes $f.bytes)) -ForegroundColor Yellow
  }
  else {
    $why = if ($have) { "present in the source tree, deliberately not shipped" } else { "absent from the source tree" }
    Write-Host ("  not staged: {0}  {1}  -- {2}" -f $f.path, (Format-Bytes $f.bytes), $why) -ForegroundColor DarkGray
  }
}

Write-Host ("  weights: {0}/{1} required files staged, {2}" -f $Staged.Count, $Required.Count, (Format-Bytes $WeightsBytes))
foreach ($f in $Optional) {
  $src = Join-Path $ModelSource ($f.path -replace '/', '\')
  if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
    Write-Host ("  not staged (optional): {0}  {1}  -- deliberate, it is not in the bundle" -f $f.path, (Format-Bytes $f.bytes)) -ForegroundColor DarkGray
  }
}

if ($Missing.Count -gt 0 -or $BadHash.Count -gt 0) {
  $msg = @()
  if ($Missing.Count -gt 0) { $msg += "missing: " + ($Missing -join ", ") }
  if ($BadHash.Count -gt 0) { $msg += "wrong: " + ($BadHash -join "; ") }
  $detail = $msg -join "`n  "
  if ($AllowMissingWeights) {
    Write-Warning "Weights incomplete and -AllowMissingWeights was passed. THIS BUILD IS NOT SHIPPPABLE.`n  $detail"
  }
  else {
    throw @"
Model weights do not match the registry, so the distribution would be broken:

  $detail

Source searched: $ModelSource

Fix it by re-running the frozen exporter, or point -ModelSource at a directory
that holds the export:

  python scripts\export_sravaani_onnx.py --src <sravaani-1.0 dir> --out artifacts\sravaani

For a local smoke test of the binaries alone, pass -AllowMissingWeights. That
switch also suppresses the installer and the checksum manifest, because an
artefact that cannot transcribe must not be signed and shipped.
"@
  }
}

# --- runtime --------------------------------------------------------------
# DLLs FLAT in runtime\. cuDNN 9's cudnn64_9.dll is a dispatcher that
# LoadLibrary's its ten siblings by bare name, and ONNX Runtime resolves
# onnxruntime.dll from the executable's own directory first. Nesting either
# breaks GPU acceleration *silently* -- the app falls back to the CPU provider
# with no error at all.
$RuntimeDlls = [System.Collections.Generic.List[object]]::new()
if (Test-Path -LiteralPath $RuntimeSource) {
  $all = Get-ChildItem -LiteralPath $RuntimeSource -Filter "*.dll" -File
  if ($RuntimeFlavour -eq "Cpu") {
    foreach ($name in $CpuRuntimeDlls) {
      $d = $all | Where-Object { $_.Name -eq $name }
      if (-not $d) { throw "CPU runtime is incomplete: $RuntimeSource has no $name. Run the frozen scripts\fetch_runtime.ps1." }
      $RuntimeDlls.Add($d)
    }
  }
  else {
    foreach ($d in $all) { $RuntimeDlls.Add($d) }
  }
  foreach ($d in $RuntimeDlls) { Copy-Item -LiteralPath $d.FullName -Destination $RuntimeStage }
}
else {
  throw "No runtime at $RuntimeSource. A distribution that ships weights but not ONNX Runtime transcribes nothing. Run the frozen scripts\fetch_runtime.ps1, or pass -RuntimeSource."
}

$RuntimeBytes = ($RuntimeDlls | Measure-Object -Property Length -Sum).Sum
Write-Host ("  runtime: {0} DLLs, {1}" -f $RuntimeDlls.Count, (Format-Bytes $RuntimeBytes))

# --- licences and documentation -------------------------------------------
# Physically present, not merely referenced. `installer\licenses\NOTICE.md` names
# every component and the packager fails below if any of the named files is
# missing, so "we mention the licence" and "we ship the licence" cannot drift.
$LicenseSrc = Join-Path $RepoRoot "installer\licenses"
$Notice = Join-Path $LicenseSrc "NOTICE.md"
if (-not (Test-Path -LiteralPath $Notice)) { throw "installer\licenses\NOTICE.md is missing; a distribution must carry its third-party notices." }

$RequiredLicenses = @("NOTICE.md", "SraVaani-MIT-LICENSE.txt", "ONNX-Runtime-LICENSE.txt")
if ($RuntimeFlavour -eq "Gpu") { $RequiredLicenses += "NVIDIA-CUDA-LICENSE.txt" }
foreach ($name in $RequiredLicenses) {
  $p = Join-Path $LicenseSrc $name
  if (-not (Test-Path -LiteralPath $p)) { throw "Licence file $name is named in NOTICE.md but not present at $p." }
  Copy-Item -LiteralPath $p -Destination $LicenseStage
}
Write-Host ("  licenses: {0} files" -f $RequiredLicenses.Count)

# ONNX Runtime's own third-party notice for the code it embeds. Taken from the
# pinned zip rather than committed, because it is 344 KB of generated text that
# changes every release and belongs to Microsoft's archive, not to this repo.
$OrtZip = Get-ChildItem -LiteralPath (Join-Path $RuntimeSource ".cache") -Filter "onnxruntime-*.zip" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($OrtZip) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $z = [System.IO.Compression.ZipFile]::OpenRead($OrtZip.FullName)
  try {
    $e = $z.Entries | Where-Object { $_.FullName -like "*/ThirdPartyNotices.txt" } | Select-Object -First 1
    if ($e) {
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $LicenseStage "ONNX-Runtime-ThirdPartyNotices.txt"), $true)
      Write-Host "  licenses: ONNX-Runtime-ThirdPartyNotices.txt (from the pinned archive)"
    }
  }
  finally { $z.Dispose() }
}
else {
  Write-Warning "No pinned ONNX Runtime archive under $RuntimeSource\.cache; ONNX-Runtime-ThirdPartyNotices.txt was not extracted. It is present in the source checkout and should accompany a release."
}

Copy-Item -LiteralPath (Join-Path $RepoRoot "installer\OFFLINE-INSTALL.md") -Destination $StageDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "scripts\a12\verify-install.ps1") -Destination $StageDir

# The model registry, shipped.
#
# Without it, `verify-install.ps1` on a user's machine can check that the seven
# model files are *present* but not that they are the *right* ones: the registry
# is the only place the expected sizes and SHA-256s exist, and it lives in the
# source tree a user does not have. That is the whole point of a self-test — a
# silently-corrupt 845 MiB encoder is the right size and the wrong bytes, and it
# shows up to a user as transcription that is mysteriously slow.
#
# 6.6 KB, and it is attribution rather than a secret: it names the model, its
# licence, its upstream repository and the hash of every file beside it.
Copy-Item -LiteralPath $RegistryPath -Destination (Join-Path $StageDir "vaani-model-registry.json")

# ---------------------------------------------------------------------------
# 4. Checksum manifest over the WHOLE distribution.
#
#    `SHA256SUMS` at the top level covers the top-level artefacts (the zip and
#    the installer). That answers "is my download intact", which is not the
#    question a user has after an install. This file answers "is my
#    *installation* intact", which is the question a user has when transcription
#    is unexpectedly slow and they suspect the weights.
# ---------------------------------------------------------------------------
Write-Host "`n[3/7] Checksum manifest" -ForegroundColor Cyan
$Shippable = -not ($AllowMissingWeights -and ($Missing.Count -gt 0 -or $BadHash.Count -gt 0))

$StagedFiles = Get-ChildItem -LiteralPath $StageDir -Recurse -File |
  Sort-Object { $_.FullName.Substring($StageDir.Length).Replace('\', '/') }

$sumLines = [System.Collections.Generic.List[string]]::new()
$sumBytes = 0L
$hashStarted = Get-Date
foreach ($f in $StagedFiles) {
  $rel = $f.FullName.Substring($StageDir.Length).Replace('\', '/').TrimStart('/')
  $sumLines.Add(("{0}  {1}" -f (Get-Sha256 $f.FullName), $rel))
  $sumBytes += $f.Length
}
$hashElapsed = (Get-Date) - $hashStarted

if ($Shippable) {
  Write-Utf8NoBom -Path (Join-Path $StageDir "vaani-dist.sha256") -Lines $sumLines
}
else {
  Write-Warning "Not writing vaani-dist.sha256: this build is missing weights and is not a distribution."
}

# `vaani-dist.json` is the machine-readable half: what is in the box, which
# runtime flavour it is, and where each part's licence is. `verify-install.ps1`
# reads it; so does anyone auditing what they were given.
$distJson = [ordered]@{
  app_version    = $Version
  layout         = "exe-relative"
  runtime_flavour = $RuntimeFlavour.ToLowerInvariant()
  model_id       = $Model.id
  model_dir      = "models/$($Model.dir)"
  model_license  = "MIT"
  model_source   = "https://huggingface.co/$($Model.source.repo)"
  bundled_files  = @($Staged | Sort-Object)
  omitted_files  = @($Optional | ForEach-Object { "$($_.path)  (optional, $(Format-Bytes $_.bytes))" })
  licences       = @($RequiredLicenses | ForEach-Object { "licenses/$_" })
  note           = "Zero-token, offline out of the box. The speech model ships with the application; no account, network or Hugging Face login is required to transcribe."
} | ConvertTo-Json -Depth 4
  Write-Utf8NoBom -Path (Join-Path $StageDir "vaani-dist.json") -Lines @($distJson)

# Re-take the file list: `vaani-dist.json` was written after the sweep above.
$StagedFiles = Get-ChildItem -LiteralPath $StageDir -Recurse -File | Sort-Object FullName
$sumBytes = ($StagedFiles | Measure-Object -Property Length -Sum).Sum
Write-Host ("  {0} files, {1} uncompressed, hashed in {2:N1}s" -f $StagedFiles.Count, (Format-Bytes $sumBytes), $hashElapsed.TotalSeconds)

$ModelBytes = ($StagedFiles | Where-Object { $_.FullName -like "$ModelDir*" } | Measure-Object -Property Length -Sum).Sum
$ExeBytes = ($StagedFiles | Where-Object { $_.Extension -eq ".exe" } | Measure-Object -Property Length -Sum).Sum
# Subtract the executables too. Leaving them in made the "licences + docs" line
# report 21.9 MiB for 0.4 MiB of documents, which is the kind of wrong number
# that gets quoted in a release note.
$DocBytes = $sumBytes - $ModelBytes - $RuntimeBytes - $ExeBytes
Write-Host "  breakdown:"
Write-Host ("    model weights    {0,12}" -f (Format-Bytes $ModelBytes))
Write-Host ("    runtime ({0,-3})   {1,12}" -f $RuntimeFlavour, (Format-Bytes $RuntimeBytes))
Write-Host ("    executables      {0,12}" -f (Format-Bytes $ExeBytes))
Write-Host ("    licences + docs  {0,12}" -f (Format-Bytes $DocBytes))
Write-Host ("    total            {0,12}" -f (Format-Bytes $sumBytes))

# ---------------------------------------------------------------------------
# 5. Portable ZIP.
# ---------------------------------------------------------------------------
Write-Host "`n[4/7] Portable ZIP" -ForegroundColor Cyan
$zip = Join-Path $OutDir "$StageName.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $zip -CompressionLevel Optimal
$zipBytes = (Get-Item -LiteralPath $zip).Length
Write-Host ("  {0}  {1}" -f (Split-Path -Leaf $zip), (Format-Bytes $zipBytes))

# ---------------------------------------------------------------------------
# 6. Installer.
# ---------------------------------------------------------------------------
Write-Host "`n[5/7] Installer" -ForegroundColor Cyan
$installer = $null
if ($AllowMissingWeights) {
  Write-Warning "Skipping the installer: this build has no weights and must not be installed on a user's machine."
}
elseif (-not $SkipInstaller) {
  $iscc = $null
  $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
  if ($null -ne $cmd) { $iscc = $cmd.Source }
  if ($null -eq $iscc) {
    foreach ($p in @("C:\Program Files (x86)\Inno Setup 6\ISCC.exe", "C:\Program Files\Inno Setup 6\ISCC.exe")) {
      if (Test-Path -LiteralPath $p) { $iscc = $p; break }
    }
  }
  if ($iscc) {
    & $iscc "/DVersion=$Version" "/DStageDir=$StageDir" (Join-Path $RepoRoot "installer\vaani.iss") "/O$OutDir"
    if ($LASTEXITCODE -ne 0) { throw "ISCC.exe failed with $LASTEXITCODE." }
    $installer = Join-Path $OutDir "VaaniDesk-Setup-$Version-win-x64.exe"
    Write-Host ("  {0}  {1}" -f (Split-Path -Leaf $installer), (Format-Bytes (Get-Item -LiteralPath $installer).Length))
  }
  else {
    Write-Warning "ISCC.exe not found; no installer built. The portable ZIP is complete and is the artefact to test on a clean machine."
  }
}
else { Write-Host "  ... skipped (-SkipInstaller)" }

# ---------------------------------------------------------------------------
# 7. Top-level checksums.
# ---------------------------------------------------------------------------
Write-Host "`n[6/7] SHA256SUMS" -ForegroundColor Cyan
$sums = Join-Path $OutDir "SHA256SUMS"
$lines = foreach ($f in @($zip, $installer) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }) {
  "{0}  {1}" -f (Get-Sha256 $f), (Split-Path -Leaf $f)
}
if ($lines) { Write-Utf8NoBom -Path $sums -Lines $lines }
else { Write-Warning "No artefacts to checksum." }

# ---------------------------------------------------------------------------
# 8. Self-test.
#
# Run against the staged tree, in place, before this script reports success. A
# packaging script that has not checked its own output is a packaging script
# that reports a working distribution on the days it is broken.
# ---------------------------------------------------------------------------
Write-Host "`n[7/7] Self-test against the staged tree" -ForegroundColor Cyan
$verify = Join-Path $RepoRoot "scripts\a12\verify-install.ps1"
if ($Shippable) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File $verify -Root $StageDir -ModelsOnly
  if ($LASTEXITCODE -ne 0) { throw "verify-install.ps1 failed against the staged tree. Not reporting success." }
}
else {
  Write-Warning "Skipping the self-test: it would fail by construction without weights."
}

Write-Host ""
if ($Shippable) {
  Write-Host "Release staging complete." -ForegroundColor Green
  Write-Host ("  distribution : {0}" -f (Format-Bytes $sumBytes))
  Write-Host ("  portable zip : {0}" -f (Format-Bytes $zipBytes))
  Write-Host "  next         : extract the zip to a directory outside any git checkout and run"
  Write-Host "                 .\vaani-cli.exe doctor --json, then .\verify-install.ps1"
}
else {
  Write-Host "Staging complete, but this is NOT a shippable distribution (no model weights)." -ForegroundColor Yellow
}

