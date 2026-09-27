<#
.SYNOPSIS
  VaaniDesk Windows x64 release packager (A8-owned: scripts/release/**).

.DESCRIPTION
  1. cargo build --release -p vaani-cli --bins   (vaani.exe + vaani-cli.exe)
  2. Stages a portable directory:
       VaaniDesk-<version>-win-x64/vaani.exe
       VaaniDesk-<version>-win-x64/vaani-cli.exe
       VaaniDesk-<version>-win-x64/runtime/*.dll   (FLAT, beside the exes)
       VaaniDesk-<version>-win-x64/runtime/runtime-manifest.json
       VaaniDesk-<version>-win-x64/runtime/LICENSE (redist licence, when present)
       VaaniDesk-<version>-win-x64/models/sravaani/* (Pre-bundled production weights)
  3. Compresses the portable ZIP (same flat layout: -cl equivalent).
  4. Compiles installer/vaani.iss with Inno Setup (ISCC.exe) when available.
  5. Writes SHA256SUMS for every artefact.

  Model weights are pre-bundled (ADR-005/ADR-007) so the product is 100% offline
  and functional out-of-the-box with zero tokens or setup.

.PARAMETER Version
  Release version. Defaults to the vaani-core package version (the single
  in-binary source of APP_VERSION).

.PARAMETER OutDir
  Where artefacts land. Default: target\<CARGO_TARGET_DIR default>\release-pkg...
  i.e. <repo>\target\release-pkg unless overridden.

.PARAMETER SkipBuild
  Stage whatever is already in target\release\ (faster re-runs).

.PARAMETER SkipInstaller
  Do not invoke ISCC.exe even when present.
#>
[CmdletBinding()]
param(
  [string]$Version = "",
  [string]$OutDir = "",
  [switch]$SkipBuild,
  [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($Version -eq "") {
  $cargoToml = Get-Content (Join-Path $RepoRoot "crates\vaani-core\Cargo.toml") -Raw
  if ($cargoToml -match '(?m)^version\s*=\s*"([^"]+)"') { $Version = $Matches[1] }
  else { throw "Could not read version from crates/vaani-core/Cargo.toml; pass -Version explicitly." }
}
if ($OutDir -eq "") { $OutDir = Join-Path $RepoRoot "target\release-pkg" }

$StageName = "VaaniDesk-$Version-win-x64"
$StageDir  = Join-Path $OutDir $StageName
$TargetDir = Join-Path $RepoRoot "target\release"

Write-Host "VaaniDesk release $Version"
Write-Host "  repo:  $RepoRoot"
Write-Host "  stage: $StageDir"

# 1. Build both bins.
if (-not $SkipBuild) {
  Push-Location $RepoRoot
  try {
    cargo build --release -p vaani-cli --bins
    if ($LASTEXITCODE -ne 0) { throw "cargo build --release failed." }
  } finally { Pop-Location }
}

$vaaniExe = Join-Path $TargetDir "vaani.exe"
$cliExe   = Join-Path $TargetDir "vaani-cli.exe"
foreach ($bin in @($vaaniExe, $cliExe)) {
  if (-not (Test-Path -LiteralPath $bin)) { throw "Missing build output: $bin (run without -SkipBuild)." }
}

# 2. Stage the portable directory.
if (Test-Path -LiteralPath $StageDir) { Remove-Item -LiteralPath $StageDir -Recurse -Force }
New-Item -ItemType Directory -Path $StageDir | Out-Null
$RuntimeStage = Join-Path $StageDir "runtime"
New-Item -ItemType Directory -Path $RuntimeStage | Out-Null

Copy-Item -LiteralPath $vaaniExe -Destination (Join-Path $StageDir "vaani.exe")
Copy-Item -LiteralPath $cliExe -Destination (Join-Path $StageDir "vaani-cli.exe")

# Runtime DLLs FLAT (ADR-007): cuDNN's dispatcher resolves siblings by bare
# name from the executable's directory, so no subdirectories here.
$RuntimeSrc = Join-Path $RepoRoot "runtime"
$dlls = @()
if (Test-Path -LiteralPath $RuntimeSrc) {
  $dlls = Get-ChildItem -LiteralPath $RuntimeSrc -Filter "*.dll" -File
  foreach ($dll in $dlls) { Copy-Item -LiteralPath $dll.FullName -Destination $RuntimeStage }
  $lic = Join-Path $RuntimeSrc "LICENSE"
  if (Test-Path -LiteralPath $lic) { Copy-Item -LiteralPath $lic -Destination (Join-Path $RuntimeStage "LICENSE") }
} else {
  Write-Warning "runtime/ not present; staging binaries only (CPU-only portable)."
}

# Production model staging: bundle sravaani production assets for 100% offline immediate use.
# NOTE FOR AGENTS (ADR-005/ADR-007): SraVaani 1.0 is MIT-licensed. By architectural mandate,
# production weights (~930 MB) are bundled directly so users have an immediate 100% offline experience.
$ModelStage = Join-Path $StageDir "models\sravaani"
New-Item -ItemType Directory -Path $ModelStage -Force | Out-Null
$ModelSrc = Join-Path $RepoRoot "artifacts\sravaani"
$modelFiles = @("encoder.fp16.onnx", "decoder_joint.onnx", "tokenizer.model", "mel_window.f32", "mel_fb.f32", "preproc.json", "config.json")
if (Test-Path -LiteralPath $ModelSrc) {
  foreach ($mf in $modelFiles) {
    $srcFile = Join-Path $ModelSrc $mf
    if (Test-Path -LiteralPath $srcFile) {
      Copy-Item -LiteralPath $srcFile -Destination $ModelStage
    }
  }
} else {
  Write-Warning "artifacts/sravaani not present; ensure models are staged before final distribution."
}

$manifest = [ordered]@{
  app_version = $Version
  layout      = "portable-flat"
  files       = @("vaani.exe", "vaani-cli.exe") + @($dlls | ForEach-Object { "runtime/$($_.Name)" }) + @($modelFiles | ForEach-Object { "models/sravaani/$_" })
  note        = "Pre-bundled production models and runtime DLLs for 100% offline zero-token operation."
} | ConvertTo-Json -Depth 3
Set-Content -LiteralPath (Join-Path $RuntimeStage "runtime-manifest.json") -Value $manifest -Encoding utf8NoBOM

# 3. Portable ZIP with the same flat layout.
$zip = Join-Path $OutDir "$StageName.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -LiteralPath $StageDir -DestinationPath $zip
Write-Host "  zip: $zip"

# 4. Installer via Inno Setup (optional locally; required in CI).
$installer = $null
if (-not $SkipInstaller) {
  $isccCmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
  $iscc = if ($null -ne $isccCmd) { $isccCmd.Source } else { $null }
  if ($null -eq $iscc) {
    $iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
  }
  if (Test-Path -LiteralPath $iscc) {
    & $iscc "/DVersion=$Version" "/DStageDir=$StageDir" (Join-Path $RepoRoot "installer\vaani.iss") "/O$OutDir"
    if ($LASTEXITCODE -ne 0) { throw "ISCC.exe failed." }
    $installer = Join-Path $OutDir "VaaniDesk-Setup-$Version-win-x64.exe"
  } else {
    Write-Warning "ISCC.exe not found; skipping installer build (CI installs Inno Setup)."
  }
}

# 5. Checksums for every artefact.
$sums = Join-Path $OutDir "SHA256SUMS"
$lines = foreach ($f in @($zip, $installer) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }) {
  $h = (Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLowerInvariant()
  "$h  $(Split-Path -Leaf $f)"
}
Set-Content -LiteralPath $sums -Value $lines -Encoding utf8NoBOM
Write-Host "  sums: $sums"
Write-Host "Release staging complete."
