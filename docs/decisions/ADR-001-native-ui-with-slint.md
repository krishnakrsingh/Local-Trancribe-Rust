# ADR-001 — Native UI with Slint

**Status:** Accepted (not yet built)
**Date:** 2026-09-27

## Context

VaaniDesk is a Windows desktop application: drop a meeting recording, watch it
transcribe, read a speaker-attributed transcript, export it. The interactive surface is
large — a virtualised transcript list, a transport bar that tracks playback, a settings
panel that reports live hardware capability, a drag-and-drop surface, a model download
view — and it shares a process with an 887 MB ONNX model on a 4 GB GPU.

Constraints that shaped this:

- Windows only, x64, MSVC. No cross-platform requirement exists.
- The same process must hold a CUDA context and a UI event loop. A web UI in a browser or
  an Electron shell puts a second Chromium between the user and the GPU.
- Transcript rendering must stay smooth at 5,000 segments.
- No IT department, no browser download step, no webview2 bootstrapper.

## Decision

Use **Slint 1.18.1** for the entire UI, with these specific choices:

| Choice | Value | Reason |
|---|---|---|
| Renderer | `renderer-femtovg` on OpenGL | `wgpu`/`winit` would create a **second D3D12 device** alongside CUDA's. femtovg on GL does not. |
| File picker | `rfd` 0.17 | native Windows dialog; Slint has no built-in file picker |
| Drag and drop | `unstable-winit-030`, or an OLE `IDropTarget` | Slint's `DropArea` **does not receive Explorer drops on the winit backend** |
| Transcript list | `ListView` | virtualised; 5,000 segments is not a widget count, it is a viewport count |

Presentation lives only in Slint: no HTML, no webview, no second toolkit, and no
drawing or layout maths in `vaani-app`.

## Consequences

**Licensing — the one thing to settle before shipping.** Slint is
**GPL-3.0-only OR royalty-free OR commercial**. A proprietary VaaniDesk must take the
royalty-free or the commercial licence. GPL-3.0-only would make VaaniDesk itself a
derived work and force the whole source open. This has not been recorded anywhere in the
repository and is a live obligation.

`unstable-winit-030` is an unstable feature. Its API can break on any Slint minor
release, which makes the Slint version a pinned decision rather than a floating one.

`renderer-femtovg` on OpenGL is a deliberate downgrade from wgpu's modern backends. It
costs some 2D text rendering throughput and buys the absence of a second D3D12 device on a
3.3 GB GPU. On this machine that trade is obviously correct.

The `DropArea` limitation is the sharp edge: a drop target that silently does nothing is
worse than no drop target, because the user believes the file was accepted. If
`unstable-winit-030` proves unreliable, the OLE `IDropTarget` is the fallback and it must
be implemented before the drop screen ships.

**Slint 1.18.1, `rfd` 0.17 and `hf-hub` are not in `Cargo.toml`.** They are stated intent
in `[workspace.dependencies]` and docs. No member crate depends on them, so they do not
appear in `Cargo.lock` and have never been compiled here. This ADR records a decision,
not a shipped dependency.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| Electron / Tauri + web UI | a webview2 runtime dependency, a second process, and a Chromium GPU process competing with CUDA for a 3.3 GB budget |
| egui / iced | more control, but no native Windows widgets, and the transcript virtualisation would be hand-built |
| WinUI 3 / WPF | genuinely native, but two toolchains and no path to a portable cross-Rust build |
| Qt | C++ in the process, or a Python binding; neither fits the dependency policy |
| A CLI only | the product is a transcript *editor*; a CLI cannot seek playback to a word |
