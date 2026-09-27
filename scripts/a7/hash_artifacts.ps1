<#
.SYNOPSIS
    Print the JSON fragment that vaani-models' manifest needs for a directory.

.DESCRIPTION
    Hashes every file in a model directory and prints a `files: [...]` array
    ready to paste into crates/vaani-models/manifests/registry.json, plus the
    total for `size_bytes`.

    The hashes are measured from the files that are actually on disk. That is
    the whole point: artifacts/sravaani/artefacts.json is the exporter's own log
    and it is known to be stale for encoder.fp16.onnx, because it still records
    the pre-inline split into .onnx plus .onnx_data. A manifest generated from
    that log would reject a perfectly good install.

    This script only reads. It never writes a manifest, because the file list is
    the part a machine can produce and the licence, access level and hardware
    requirements are the parts only a person should.

.PARAMETER Path
    The model directory, e.g. artifacts\sravaani.

.PARAMETER Required
    Files that are needed to *run* the model, as opposed to files that exist only
    for comparison. These get "required": true and everything else false, which
    mirrors the manifest's own flag and is what the app downloads.

.EXAMPLE
    powershell -File scripts\a7\hash_artifacts.ps1 -Path artifacts\sravaani
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Path,

    [string[]] $Required = @(
        'encoder.fp16.onnx',
        'decoder_joint.onnx',
        'tokenizer.model',
        'mel_window.f32',
        'mel_fb.f32',
        'preproc.json',
        'config.json'
    )
)

$ErrorActionPreference = 'Stop'

$dir = Resolve-Path -LiteralPath $Path
$files = @(Get-ChildItem -LiteralPath $dir -File | Sort-Object Name)
if ($files.Count -eq 0) {
    Write-Error "$Path contains no files."
}

$total = 0L
$entries = @()
foreach ($f in $files) {
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $f.FullName).Hash.ToLowerInvariant()
    $isRequired = if ($Required -contains $f.Name) { 'true' } else { 'false' }
    $total += $f.Length
    $entries += ('      { "path": "' + $f.Name + '", "role": "weights", "bytes": ' +
        $f.Length + ', "sha256": "' + $hash + '", "required": ' + $isRequired + ' }')
}

Write-Output ('    "size_bytes": {0},' -f $total)
Write-Output '    "files": ['
for ($i = 0; $i -lt $entries.Count; $i++) {
    $comma = if ($i -lt $entries.Count - 1) { ',' } else { '' }
    Write-Output ($entries[$i] + $comma)
}
Write-Output '    ]'
Write-Host ''
Write-Host ("# {0} bytes across {1} files, measured {2}." -f $total, $files.Count, (Get-Date -Format 'yyyy-MM-dd'))
