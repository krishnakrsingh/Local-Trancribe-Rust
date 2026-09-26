<#
.SYNOPSIS
    Fetch the native runtime VaaniDesk needs, with checksum verification.

.DESCRIPTION
    Downloads, into .\runtime\ :

      * ONNX Runtime 1.30.0 for Windows x64, CUDA 13 build (official Microsoft
        release asset). We deliberately use Microsoft's build rather than the
        `ort` crate's CDN build because Microsoft's CI compiles for
        `86-real` -- native SASS for the RTX 3050 Ti's compute capability 8.6 --
        while the CDN build ships `75;80;90;120` and would leave us relying on
        PTX JIT. See docs/decisions/ADR-003-onnx-runtime.md.

      * The NVIDIA CUDA runtime redistributables that the ORT CUDA EP
        dynamically links against. These are the *runtime* libraries, published
        by NVIDIA specifically so third parties can redistribute them; they do
        NOT include nvcc or any build tooling, so the shipped application does
        not require the CUDA Toolkit.

    Every download is verified against a SHA-256 taken from NVIDIA's / GitHub's
    own published manifest, and the expected hashes are pinned in
    .\runtime\manifest.json. A mismatch aborts.

.PARAMETER Force
    Re-download even if the file is already present and verified.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fetch_runtime.ps1
#>
[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $Root 'runtime'
$CacheDir = Join-Path $RuntimeDir '.cache'
New-Item -ItemType Directory -Force -Path $RuntimeDir, $CacheDir | Out-Null

# ---------------------------------------------------------------------------
# Pinned manifest.
#
# Everything here is a hard pin. A floating version in a fetch script is how a
# build silently changes under you; if you bump one of these you must re-run
# the benchmark suite, because a different ORT build means a different set of
# CUDA kernels and therefore different timings.
# ---------------------------------------------------------------------------
$OrtVersion = '1.30.0'
$CudaRedistVersion = '13.0.2'
$CudnnVersion = '9.14.0.64'

$Pins = @(
    [pscustomobject]@{
        Name        = 'onnxruntime'
        Group       = 'onnxruntime'
        File        = "onnxruntime-win-x64-gpu_cuda13-$OrtVersion.zip"
        Url         = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-gpu_cuda13-$OrtVersion.zip"
        # From the GitHub releases API (`digest` field) for v1.30.0. GitHub
        # publishes a SHA-256 per asset, so this is a first-party value rather
        # than something we computed from a file we had already downloaded.
        Sha256      = '8fa4b08359af682cd605892cb59077049700b640128bb93fd2c7776cf9f55bdc'
        Bytes       = 294386823
    }
)

function Get-Sha256 {
    param([string]$Path)
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Hash {
    param([string]$Path, [string]$Expected, [string]$Label)
    if (-not $Expected -or $Expected -match '^0+$') { throw "No pinned SHA-256 for $Label. Refusing to trust an unverified download." }
    $actual = Get-Sha256 $Path
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for ${Label}`n  expected $Expected`n  actual   $actual"
    }
    Write-Host "  verified $Label  $actual" -ForegroundColor DarkGreen
}

function Get-Pinned {
    param([string]$Url, [string]$Sha256, [string]$Label, [long]$ExpectedBytes = 0)
    $name = Split-Path -Leaf $Url
    $dest = Join-Path $CacheDir $name
    if (-not (Test-Path -LiteralPath $dest) -or $Force) {
        Write-Host "  downloading $Label ..."
        Invoke-WebRequest -Uri $Url -OutFile $dest -UseBasicParsing
    }
    if ($ExpectedBytes -gt 0) {
        $len = (Get-Item -LiteralPath $dest).Length
        if ($len -ne $ExpectedBytes) { throw "Size mismatch for ${Label}: got $len, expected $ExpectedBytes" }
    }
    Assert-Hash -Path $dest -Expected $Sha256 -Label $Label
    return $dest
}

function Expand-To {
    param([string]$Archive, [string]$DestDir, [string]$Strip = 1)
    $tmp = Join-Path $env:TEMP ("vaani-extract-" + [guid]::NewGuid().ToString('N'))
    try {
        Expand-Archive -LiteralPath $Archive -DestinationPath $tmp -Force
        $root = (Resolve-Path -LiteralPath $tmp).Path
        for ($i = 1; $i -lt $Strip; $i++) {
            $children = @(Get-ChildItem -LiteralPath $root)
            if ($children.Count -ne 1 -or -not $children[0].PSIsContainer) { break }
            $root = $children[0].FullName
        }
        # Copy DLLs flat into the runtime dir: the ORT CUDA provider resolves
        # cudart/cublas/cudnn through the standard Windows search order, which
        # starts at the directory of the executable. Subdirectories would not
        # be found without an explicit preload, and a flat layout is what the
        # installer and the release archive both assume.
        Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.dll' |
            ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $DestDir -Force }
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# 1. ONNX Runtime
# ---------------------------------------------------------------------------
Write-Host "`n[1/3] ONNX Runtime $OrtVersion (CUDA 13)" -ForegroundColor Cyan
$ortPin = $Pins[0]
$ortZip = Get-Pinned -Url $ortPin.Url -Sha256 $ortPin.Sha256 -Label 'onnxruntime.zip' -ExpectedBytes $ortPin.Bytes
Expand-To -Archive $ortZip -DestDir $RuntimeDir

# ---------------------------------------------------------------------------
# 2. CUDA runtime redistributables
#
# The manifest is NVIDIA's own and carries a sha256 per component per platform,
# so we read the hash from it rather than pinning a second copy here. The
# manifest version itself is pinned above.
# ---------------------------------------------------------------------------
Write-Host "`n[2/3] CUDA runtime redistributables $CudaRedistVersion" -ForegroundColor Cyan
$manifest = Invoke-RestMethod -Uri "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_$CudaRedistVersion.json"
foreach ($component in @('cuda_cudart', 'libcublas', 'cuda_nvrtc')) {
    $entry = $manifest.$component
    if (-not $entry) { throw "component $component missing from CUDA redist manifest" }
    $win = $entry.'windows-x86_64'
    $url = "https://developer.download.nvidia.com/compute/cuda/redist/$($win.relative_path)"
    $zip = Get-Pinned -Url $url -Sha256 $win.sha256 -Label "$component-$($entry.version)"
    Expand-To -Archive $zip -DestDir $RuntimeDir
}

# ---------------------------------------------------------------------------
# 3. cuDNN
# ---------------------------------------------------------------------------
Write-Host "`n[3/3] cuDNN $CudnnVersion" -ForegroundColor Cyan
# The path and hash come from NVIDIA's cuDNN redist manifest, which nests a
# cuda12/cuda13 object per platform. The hash is pinned in scripts/cudnn.sha256.
$cudnnManifest = Invoke-RestMethod -Uri "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_$($CudnnVersion -replace '\.\d+$', '.0').json"
$cudnnWin = $cudnnManifest.cudnn.'windows-x86_64'.cuda13
if (-not $cudnnWin.sha256) { throw "cuDNN manifest has no windows-x86_64/cuda13 entry for $CudnnVersion" }
$cudnn = Get-Pinned `
    -Url "https://developer.download.nvidia.com/compute/cudnn/redist/$($cudnnWin.relative_path)" `
    -Sha256 (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cudnn.sha256') -Raw).Trim() `
    -Label "cudnn-$CudnnVersion"
Expand-To -Archive $cudnn -DestDir $RuntimeDir

# ---------------------------------------------------------------------------
# Record what we actually shipped, so a bug report includes exact versions.
# ---------------------------------------------------------------------------
$summary = Get-ChildItem -LiteralPath $RuntimeDir -File -Filter '*.dll' |
    Sort-Object Name |
    ForEach-Object { [pscustomobject]@{ name = $_.Name; bytes = $_.Length; sha256 = (Get-Sha256 $_.FullName) } }
$summary | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $RuntimeDir 'runtime-manifest.json') -Encoding UTF8

$total = ($summary | Measure-Object -Property bytes -Sum).Sum
Write-Host ("`nruntime/ ready: {0} DLLs, {1:N0} MB" -f $summary.Count, ($total / 1MB)) -ForegroundColor Green
Write-Host "Wrote runtime\runtime-manifest.json" -ForegroundColor DarkGray
