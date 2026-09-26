# ADR-008 — WinGet distribution

**Status:** Accepted (not yet built)
**Date:** 2026-09-27

## Context

GitHub Releases (ADR-007) gives a verifiable download and an update channel, but it does
not put VaaniDesk in `winget search`, and installing by hand means decompressing a ZIP
somewhere and finding the executable. On Windows 11, `winget` is preinstalled, and the
Package Manager is the discovery path most non-technical users already trust.

The constraint that matters: an installer that puts `runtime/` anywhere other than beside
the executable breaks GPU acceleration, and the user gets a working product running 20–40×
slower with no error message ([COMMON_FAILURES §1](../agent/COMMON_FAILURES.md),
[COMMON_FAILURES §4](../agent/COMMON_FAILURES.md)).

## Decision

Ship a **WinGet package** in addition to the GitHub Releases ZIP, with the WinGet manifest
**pointing at the same GitHub Releases asset URL**. One artefact, two channels.

| Field | Value |
|---|---|
| Package identifier | `VaaniDesk.VaaniDesk` (to be confirmed against the manifest namespace) |
| Manifest repository | a public repo in the `microsoft/winget-pkgs` community repository |
| Installer type | EXE or MSI, produced by a build we own |
| `InstallerSha256` | the hash of the **installer**, published by the release pipeline |
| Install scope | per-machine, so `runtime/` can sit under `Program Files` beside the binaries |
| `InstallLocation` | a fixed path with `runtime\` as a **subdirectory of the same root, all DLLs flat inside it** |

The layout requirement is the load-bearing part: `vaani.exe`, `vaani-cli.exe` and
`runtime\{onnxruntime.dll, onnxruntime_providers_cuda.dll, cudart64_13.dll, cublas64_13.dll,
cudnn64_9.dll, cudnn_engines_runtime_compiled64_9.dll, ...}` — the cuDNN sibling must be a
sibling, not nested.

`RuntimeLocation::discover` handles this shape: it looks for `<exe>/runtime` first and sets
`portable_layout = true` (`runtime.rs:118-121`). If `runtime/` were nested one level
deeper, or the DLLs placed in per-component subdirectories, cuDNN's dispatcher would fail
to resolve its siblings and acceleration would silently fall back to the CPU.

`vaani doctor` is the verification surface. `RuntimeReport::collect()` never returns an
error (`lib.rs:61-75`) precisely so it can run on a machine where the install is broken and
say which DLLs are missing. It should be the first thing the installer and the updater
check.

## Consequences

WinGet gives `winget upgrade` for free, which means the in-app `vaani-updater` has a
competitor on the same machine. Two upgraders is a real hazard: both may try to replace the
same files while the other is running. The app must detect a pending WinGet upgrade and
defer, or must be a no-op when `vaani-updater` has already handled it. This needs deciding
before the first release that ships both.

A WinGet manifest is a **pull request against a public repository** from a Microsoft-verified
or community publisher account. That is an account, a review queue, and an
`InstallerSha256` regenerated on every release. The manifest is a second place where a
version number lives, alongside `Cargo.toml` and the GitHub Release tag, and all three must
agree. `APP_VERSION` (`crates/vaani-core/src/lib.rs:53`) is the single in-binary source; the
other two are generated from the release tag.

WinGet is also a compatibility surface: `winget` imposes installation requirements —
architecture, minimum OS build, and administrator-vs-user scope rules. This is a Windows 11
x64 product (MSVC host, `x86-64-pc-windows-msvc`), so the manifest must declare x64 and the
minimum OS build explicitly rather than inheriting a default.

None of this exists yet. There is no installer, no manifest, no `.github/`, and no
`vaani.exe`. The package identifier, the publisher account and the minimum-OS declaration are
all still open.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| Microsoft Store / MSIX only | store review per release, and MSIX sandboxing fights a 1.25 GiB native `runtime/` beside the executable |
| Winget only, no ZIP | `winget` is absent on Windows LTSC and in some managed environments; the ZIP is the floor |
| An auto-updating installer only (Inno Setup / Squirrel) | reinvents `winget upgrade` and can corrupt a running install |
| Chocolatey | smaller reach than WinGet on Windows 11, and a second manifest for the same installer |
| Bundling model weights in the installer | gated artefacts, and 2.7 GB more before the app can do anything (ADR-005) |
