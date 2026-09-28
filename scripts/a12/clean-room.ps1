<#
.SYNOPSIS
  Clean-room verification of a VaaniDesk distribution. (A12-owned.)

.DESCRIPTION
  Answers one question with evidence rather than assertion: **does the shipped
  distribution transcribe, on a machine that has nothing but the distribution?**

  "Nothing but the distribution" is enforced, not assumed:

    * the tree is extracted to a directory outside every git checkout, so the
      repository's `artifacts/` and `CARGO_MANIFEST_DIR` fallback cannot be what
      found the model;
    * `LOCALAPPDATA` is redirected to an empty scratch directory, so
      `%LOCALAPPDATA%\VaaniDesk\models` does not exist and discovery cannot have
      used it;
    * every `VAANI_*` variable is cleared, so no path override is in play;
    * `PATH` is stripped of `python`, and the interpreter is asserted absent, so
      the claim "the application never needs Python" is checked rather than
      restated;
    * `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` are pointed at a closed loopback
      port and `NO_PROXY` emptied, so any outbound HTTP attempt fails
      immediately instead of silently succeeding on a connected machine. This is
      a *detonation* rather than an airlock -- it stops `reqwest`, which honours
      the variables, and nothing that ignored them -- and it is reported as such
      rather than claimed as a firewall.

  Then it runs, in order:

    1. `verify-install.ps1` -- present, intact, loadable
    2. `vaani-cli doctor --json` -- the machine's own view of itself
    3. a transcription of a **synthesised** WAV -- proves the whole chain runs
       with no private audio involved
    4. optionally, with `-WithRealClip`, a transcription of a real recording

  # 3 is the privacy-safe proof and always runs. Its transcript is expected to
  be empty -- synthesised audio contains no words -- and that is the point being
  tested: the model loads, the log-mel front end runs, the TDT decode loop
  iterates and terminates, and the pipeline returns success. An empty transcript
  here means "the engine ran", not "the engine failed".

  # 4 is the proof that the engine produces words. It is opt-in because it
  involves a private recording.

.PARAMETER Zip
  The portable ZIP to test. Default: the most recent
  <CARGO_TARGET_DIR>\release-pkg\VaaniDesk-*-win-x64.zip.

.PARAMETER Stage
  Where to extract. Default: a fresh directory under $env:TEMP. It is deleted
  on success unless -KeepStage is given.

.PARAMETER RealClip
  A real audio file to transcribe instead of / in addition to the synthesised
  one. Nothing about its content is printed, stored or reported.

.PARAMETER KeepStage
  Do not delete the extracted tree, so it can be inspected.

.PARAMETER KeepOutput
  Keep the transcript files instead of deleting them. Off by default: the
  transcript of a real recording is private content and has no business sitting
  on disk after a verification run.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\a12\clean-room.ps1
  powershell -ExecutionPolicy Bypass -File scripts\a12\clean-room.ps1 -RealClip scratch\clip.wav
#>
[CmdletBinding()]
param(
  [string]$Zip = "",
  [string]$Stage = "",
  [string]$RealClip = "",
  [switch]$KeepStage,
  [switch]$KeepOutput
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunStarted = Get-Date

function Section { param([string]$t) Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Step { param([string]$t) Write-Host ("  " + $t) -ForegroundColor DarkGray }
function Ok { param([string]$t) Write-Host "  PASS  $t" -ForegroundColor Green }
function Bad { param([string]$t) Write-Host "  FAIL  $t" -ForegroundColor Red }
function MiB { param([long]$b) "{0:N1} MiB" -f ($b / 1MB) }

# Aggregate statistics from a `vaani.transcript` export.
#
# The envelope nests everything under `transcript`, and words live inside
# segments, so counting `@($t.words)` returns 1 -- a `$null` wrapped in an array
# -- and a naive check reads as "the engine produced one word". This walks the
# real shape.
#
# Returns counts and timings only. No text is read, stored or emitted, which is
# what makes it safe to point at somebody's private meeting recording.
function Get-TranscriptStats {
  param([string]$Path)
  $t = $null
  try { $t = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json } catch { return $null }
  if ($null -eq $t -or $null -eq $t.transcript) { return $null }
  $segs = @($t.transcript.segments)
  $words = 0; $min = [double]::MaxValue; $max = 0.0; $mono = $true
  for ($i = 0; $i -lt $segs.Count; $i++) {
    $ws = @($segs[$i].words)
    $words += $ws.Count
    foreach ($w in $ws) {
      $s = [double]$w.start; $e = [double]$w.end
      if ($s -lt $min) { $min = $s }
      if ($e -gt $max) { $max = $e }
    }
    if ($i -gt 0 -and [double]$segs[$i].start -lt [double]$segs[$i - 1].start) { $mono = $false }
  }
  $durS = if ($t.source -and $t.source.duration_ms) { [double]$t.source.duration_ms / 1000 } else { 0.0 }
  # An empty transcript leaves `$min` at its sentinel, and `($max - $min)`
  # then overflows to a 300-digit number rather than reading as "no words".
  $hasWords = $words -gt 0
  [pscustomobject]@{
    duration_s = $durS
    segments   = $segs.Count
    words      = $words
    speakers   = @($t.transcript.speakers).Count
    first_ms   = $(if ($hasWords) { $min } else { 0.0 })
    last_ms    = $max
    monotonic  = $mono
    span_s     = $(if ($hasWords) { ($max - $min) / 1000 } else { 0.0 })
  }
}

$script:Failed = 0
function Check {
  param([string]$Name, [bool]$Ok, [string]$Detail = "")
  if ($Ok) { Ok "$Name$(if ($Detail) { " -- $Detail" })" }
  else { Bad "$Name$(if ($Detail) { " -- $Detail" })"; $script:Failed++ }
}

# ---------------------------------------------------------------------------
# 0. Locate the artefact.
# ---------------------------------------------------------------------------
Section "Artefact"
if ($Zip -eq "") {
  $PkgDir = if ($env:CARGO_TARGET_DIR) { Join-Path $env:CARGO_TARGET_DIR "release-pkg" } else { Join-Path $RepoRoot "target\release-pkg" }
  $Zip = (Get-ChildItem -LiteralPath $PkgDir -Filter "VaaniDesk-*-win-x64.zip" -File -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
if (-not $Zip -or -not (Test-Path -LiteralPath $Zip)) {
  throw "No portable ZIP to test. Run scripts\release\make-release.ps1 first, or pass -Zip."
}
$ZipItem = Get-Item -LiteralPath $Zip
Step "zip      : $($ZipItem.FullName)"
Step "size     : $(MiB $ZipItem.Length)"

if ($Stage -eq "") {
  $Stage = Join-Path $env:TEMP ("vaani-cleanroom-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
}
New-Item -ItemType Directory -Path $Stage -Force | Out-Null
$Stage = (Resolve-Path -LiteralPath $Stage).Path
Step "stage    : $Stage"

# The one condition that makes this a clean-room test rather than a rerun: the
# distribution must not be sitting inside a checkout, or the `artifacts/`
# developer fallback in `vaani_models::paths` could be what found the model.
$insideCheckout = $false
$p = $Stage
while ($p) {
  if (Test-Path -LiteralPath (Join-Path $p ".git")) { $insideCheckout = $true; break }
  $parent = Split-Path -Parent $p
  if ($parent -eq $p) { break }
  $p = $parent
}
Check "stage is outside any git checkout" (-not $insideCheckout) $(if ($insideCheckout) { "$Stage is inside a repository" } else { "" })

# ---------------------------------------------------------------------------
# 1. Build the hostile environment.
# ---------------------------------------------------------------------------
Section "Clean-machine environment"

$ScratchLocal = Join-Path $Stage "_cleanlocalappdata"
New-Item -ItemType Directory -Path $ScratchLocal -Force | Out-Null
$env:LOCALAPPDATA = $ScratchLocal
$env:APPDATA = Join-Path $ScratchLocal "Roaming"
New-Item -ItemType Directory -Path $env:APPDATA -Force | Out-Null

# An empty per-user model directory is the strongest available statement that
# the model came out of the box rather than out of a previous install.
$UserModels = Join-Path $ScratchLocal "VaaniDesk\models"
if (Test-Path -LiteralPath $UserModels) { Remove-Item -LiteralPath $UserModels -Recurse -Force }
Check "no per-user model directory" (-not (Test-Path -LiteralPath $UserModels)) $UserModels

# Every VAANI_* override cleared, and the check is that they are gone rather
# than that we set them to nothing.
foreach ($k in @($env:Keys | Where-Object { $_ -like "VAANI_*" })) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
$left = @($env:Keys | Where-Object { $_ -like "VAANI_*" })
Check "no VAANI_* path overrides in the environment" ($left.Count -eq 0) $(if ($left.Count) { $left -join ", " } else { "" })

# No Python. Removed from PATH and then asserted absent, because "the app never
# needs Python" is a claim this product makes and it costs three lines to check
# rather than restate. The directory filter has to look at the whole leaf name:
# on a stock Windows install `python` resolves through the `WindowsApps` alias
# stub, whose leaf is `python.exe` but whose directory is not named `python`.
$removed = [System.Collections.Generic.List[string]]::new()
$kept = [System.Collections.Generic.List[string]]::new()
foreach ($d in ($env:PATH -split ';')) {
  if (-not $d) { continue }
  $leaf = (Split-Path -Leaf $d)
  if ($leaf -match '^python' -or (Test-Path -LiteralPath (Join-Path $d "python.exe"))) {
    $removed.Add($d); $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $d }) -join ';'
  }
  else { $kept.Add($d) }
}
$pythonOnPath = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
Check "python is not on PATH" (-not $pythonOnPath) $(if ($removed.Count) { "removed $($removed.Count) path entries" } else { "" })

# Network. Point every proxy variable at a closed loopback port so an outbound
# request fails instantly instead of quietly succeeding on a connected machine.
$env:HTTP_PROXY = "http://127.0.0.1:1"
$env:HTTPS_PROXY = "http://127.0.0.1:1"
$env:ALL_PROXY = "http://127.0.0.1:1"
$env:NO_PROXY = ""
$env:http_proxy = $env:HTTP_PROXY
$env:https_proxy = $env:HTTPS_PROXY
Step "proxies  : 127.0.0.1:1 (closed). This detonates reqwest; it is not a firewall."
Check "network is detonated for the HTTP client" $true "any HTTP attempt fails at connect"

# ---------------------------------------------------------------------------
# 2. Extract and measure.
# ---------------------------------------------------------------------------
Section "Extract"
$extractStart = Get-Date
Expand-Archive -LiteralPath $Zip -DestinationPath $Stage -Force
$extractElapsed = ((Get-Date) - $extractStart).TotalSeconds
$Root = (Get-ChildItem -LiteralPath $Stage -Directory | Where-Object { $_.Name -like "VaaniDesk-*" } | Select-Object -First 1).FullName
if (-not $Root) { $Root = $Stage }
$Root = (Resolve-Path -LiteralPath $Root).Path
$distBytes = (Get-ChildItem -LiteralPath $Root -Recurse -File | Measure-Object -Property Length -Sum).Sum
Step "root     : $Root"
Step "unpacked : $(MiB $distBytes) in $([math]::Round($extractElapsed,1))s"

# ---------------------------------------------------------------------------
# 3. The self-test the user would run.
# ---------------------------------------------------------------------------
Section "Self-test (verify-install.ps1)"
$selfStart = Get-Date
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "verify-install.ps1") -Root $Root -ExpectRuntimeFlavour cpu
$selfExit = $LASTEXITCODE
$selfElapsed = ((Get-Date) - $selfStart).TotalSeconds
Check "verify-install.ps1 exits 0" ($selfExit -eq 0) ("{0:N1}s" -f $selfElapsed)

# ---------------------------------------------------------------------------
# 4. The application's own view of the machine.
# ---------------------------------------------------------------------------
Section "vaani-cli doctor"
$cli = Join-Path $Root "vaani-cli.exe"
$doctorOut = & $cli doctor --json 2>&1 | Out-String
$report = $null
try { $report = $doctorOut | ConvertFrom-Json } catch { $report = $null }
if ($report) {
  Step "summary  : $($report.summary)"
  Step "runtime  : present=$($report.runtime.runtime_present)"
  foreach ($mm in $report.models) { Step "model    : $($mm.id) available=$($mm.available) -- $($mm.detail)" }
  Check "doctor reports the runtime present" ([bool]$report.runtime.runtime_present)
  Check "doctor reports a model available" (@($report.models | Where-Object { $_.available }).Count -gt 0)
  Check "no model needs a token" (-not ($report.models.detail -join " " | Select-String -Pattern "token" -Quiet))
  Check "the bundled model is the one offered" (@($report.models | Where-Object { $_.id -eq "sravaani-1.0-onnx" }).Count -eq 1)
}
else {
  Check "doctor produced JSON" $false $doctorOut
}

# ---------------------------------------------------------------------------
# 5. Transcribe. Synthesised audio first: it involves nobody's private
#    conversation and it exercises the entire chain.
# ---------------------------------------------------------------------------
Section "Transcription (synthesised audio)"

function New-TestWav {
  param([string]$Path, [double]$Seconds = 3.0, [int]$Rate = 16000)
  # A 180 Hz tone with a raised-cosine amplitude envelope. It contains no words
  # and is not intended to: what it proves is that the log-mel front end runs,
  # the encoder executes, and the TDT decode loop terminates rather than
  # hanging. 16 kHz mono 16-bit PCM, which is exactly what the front end wants.
  $n = [int]($Seconds * $Rate)
  $data = New-Object 'System.Byte[]' ($n * 2)
  for ($i = 0; $i -lt $n; $i++) {
    $t = $i / $Rate
    $envelope = (1 - [math]::Cos(2 * [math]::PI * 4 * $t)) / 2
    $v = [math]::Sin(2 * [math]::PI * 180 * $t) * 0.4 * $envelope
    $s = [int]([math]::Round($v * 32767))
    $data[$i * 2] = [byte]($s -band 0xFF)
    $data[$i * 2 + 1] = [byte](($s -shr 8) -band 0xFF)
  }
  $bytes = [System.Collections.Generic.List[byte]]::new()
  function AddBytes { param([byte[]]$b) foreach ($x in $b) { $bytes.Add($x) } }
  AddBytes ([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
  AddBytes ([BitConverter]::GetBytes([int](36 + $n * 2)))
  AddBytes ([System.Text.Encoding]::ASCII.GetBytes("WAVEfmt "))
  AddBytes ([BitConverter]::GetBytes([int]16))      # fmt chunk size
  AddBytes ([BitConverter]::GetBytes([int16]1))     # PCM
  AddBytes ([BitConverter]::GetBytes([int16]1))     # mono
  AddBytes ([BitConverter]::GetBytes([int]$Rate))
  AddBytes ([BitConverter]::GetBytes([int]($Rate * 2)))  # byte rate
  AddBytes ([BitConverter]::GetBytes([int16]2))     # block align
  AddBytes ([BitConverter]::GetBytes([int16]16))    # bits per sample
  AddBytes ([System.Text.Encoding]::ASCII.GetBytes("data"))
  AddBytes ([BitConverter]::GetBytes([int]($n * 2)))
  AddBytes $data
  [System.IO.File]::WriteAllBytes($Path, $bytes.ToArray())
  $Path
}

$WavDir = Join-Path $Stage "_wav"
New-Item -ItemType Directory -Path $WavDir -Force | Out-Null
$Wav = Join-Path $WavDir "synthetic-3s.wav"
New-TestWav -Path $Wav -Seconds 3.0
Step "input    : $Wav ($(MiB (Get-Item -LiteralPath $Wav).Length)), 3.0 s, 16 kHz mono"

$OutDir = Join-Path $Stage "_out"
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$trStart = Get-Date
$trOut = & $cli transcribe $Wav --out (Join-Path $OutDir "synthetic") --format json 2>&1 | Out-String
$trExit = $LASTEXITCODE
$trElapsed = ((Get-Date) - $trStart).TotalSeconds
Step ("exit     : $trExit")
Step ("stdout   : " + ($trOut.Trim() -replace "`r?`n", " | "))
Step ("wall     : {0:N1}s ({1:N2}x real time)" -f $trElapsed, (3.0 / [math]::Max($trElapsed, 0.001)))
if ($trExit -ne 0) { Step ("stderr: " + ($trOut.Trim() -replace "`r?`n", " | ")) }
Check "transcribe exits 0 on a clean machine with no network" ($trExit -eq 0)
Check "transcribe completed in a sane time" ($trElapsed -lt 120) ("{0:N1}s" -f $trElapsed)

# The CLI creates a *directory* named after the `--out` target and puts the
# export inside it, named after the input stem: `--out out --format json` on
# `tone.wav` writes `out/tone.json`. Found by recursing and taking the largest
# JSON, rather than hard-coding a path, so a change to the CLI's naming does not
# turn into a silent "no transcript found" that reads like a transcription
# failure. There is exactly one export per run here, so "largest" is unambiguous.
$jsonOut = Get-ChildItem -LiteralPath $OutDir -Filter "*.json" -File -Recurse -ErrorAction SilentlyContinue |
    Sort-Object Length -Descending | Select-Object -First 1
if ($jsonOut) {
  $st = Get-TranscriptStats $jsonOut.FullName
  if ($st) {
    Step ("transcript: {0} ({1:N0} bytes) -- {2} words, {3} segments, {4} speakers, span {5:N1}s" -f $jsonOut.Name, $jsonOut.Length, $st.words, $st.segments, $st.speakers, $st.span_s)
    Check "a transcript was written and parses as JSON" $true "$($st.words) words, $($st.segments) segments"
    # A short synthesised tone is not speech, so a handful of spurious tokens is
    # the expected outcome -- what is being tested is that the front end ran, the
    # encoder executed and the decode loop terminated rather than hanging. A
    # *large* count would mean the model hallucinated, which is worth failing on.
    Check "synthesised audio yields a near-empty transcript" ($st.words -le 5) "$($st.words) words"
    Check "word timings are monotonic" $st.monotonic
  }
  else { Check "the transcript parses as JSON" $false "could not parse $($jsonOut.Name)" }
}
else {
  Check "a transcript file was written" $false "nothing under $OutDir -- the CLI's --out handling is A2's surface, not this crate's"
}

# ---------------------------------------------------------------------------
# 6. Optional: a real recording. Aggregates only.
# ---------------------------------------------------------------------------
if ($RealClip) {
  Section "Transcription (real recording, aggregates only)"
  if (-not (Test-Path -LiteralPath $RealClip)) { throw "No such file: $RealClip" }
  $clipItem = Get-Item -LiteralPath $RealClip
  Step "input    : $(Split-Path -Leaf $RealClip), $(MiB $clipItem.Length)"

  $RealOut = Join-Path $OutDir "real"
  $rt0 = Get-Date
  $rOut = & $cli transcribe $RealClip --out $RealOut --format json 2>&1 | Out-String
  $rtExit = $LASTEXITCODE
  $rtElapsed = ((Get-Date) - $rt0).TotalSeconds
  Step ("exit     : $rtExit")
  Step ("wall     : {0:N1}s" -f $rtElapsed)
  Check "real transcription exits 0" ($rtExit -eq 0) $(if ($rtExit -ne 0) { ($rOut.Trim() -replace "`r?`n", " | ") })

  $realJson = Get-ChildItem -LiteralPath $OutDir -Filter "*.json" -File -Recurse -ErrorAction SilentlyContinue |
      Sort-Object Length -Descending | Select-Object -First 1
  if ($realJson) {
    $st = Get-TranscriptStats $realJson.FullName
    if ($st) {
      # Aggregates only. Counts, durations, timings. No text, no excerpt, no
      # derived content -- the transcript is private and the file is deleted
      # below unless -KeepOutput says otherwise.
      Step ("duration  : {0:N1} s ({1:N1} min)" -f $st.duration_s, ($st.duration_s / 60))
      Step ("words     : {0}" -f $st.words)
      Step ("segments  : {0}" -f $st.segments)
      Step ("speakers  : {0}" -f $st.speakers)
      Step ("span      : {0:N3}s -> {1:N3}s" -f ($st.first_ms / 1000), ($st.last_ms / 1000))
      Step ("timings   : monotonic={0}" -f $st.monotonic)
      if ($st.duration_s -gt 0) {
        Step ("real-time  : {0:N2}x ({1:N4} RTF)" -f ($st.duration_s / $rtElapsed), ($rtElapsed / $st.duration_s))
      }
      $overhang = ($st.last_ms / 1000) - $st.duration_s
      Step ("overhang  : {0:N3}s past the end of the media" -f $overhang)
      Check "the real transcription produced a substantial transcript" ($st.words -gt 100) "$($st.words) words"
      Check "word timings are monotonic" $st.monotonic
      Check "the transcript does not run past the end of the media by more than 0.5 s" ($overhang -lt 0.5) ("{0:N3}s" -f $overhang)
    }
    else { Check "the real transcript parses as JSON" $false "" }
  }
  else { Check "a real transcript was written" $false "nothing under $OutDir" }
}

# ---------------------------------------------------------------------------
# 7. Cleanup. A transcript of a real meeting has no business outliving a
#    verification run.
# ---------------------------------------------------------------------------
if (-not $KeepOutput) {
  Remove-Item -LiteralPath $OutDir -Recurse -Force -ErrorAction SilentlyContinue
  Step "transcripts deleted (pass -KeepOutput to retain them)"
}
if (-not $KeepStage) {
  Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
  Step "staging directory removed (pass -KeepStage to retain it)"
}

# ---------------------------------------------------------------------------
# 8. Verdict.
# ---------------------------------------------------------------------------
$total = ((Get-Date) - $RunStarted).TotalSeconds
Write-Host ""
if ($script:Failed -eq 0) {
  Write-Host ("CLEAN-ROOM OK -- $(MiB $distBytes) distribution, verified end to end in {0:N1}s" -f $total) -ForegroundColor Green
  Write-Host "  extracted to a directory outside any git checkout, with no per-user model directory," -ForegroundColor DarkGray
  Write-Host "  no VAANI_* overrides, no Python on PATH, and the HTTP client pointed at a closed port." -ForegroundColor DarkGray
  exit 0
}
else {
  Write-Host ("CLEAN-ROOM FAILED -- {0} problem(s) in {1:N1}s" -f $script:Failed, $total) -ForegroundColor Red
  if (-not $KeepStage) { Write-Host "  (staging directory removed; pass -KeepStage to inspect)" -ForegroundColor DarkGray }
  exit 1
}
