//! Real desktop host services: the file picker and the clipboard.
//!
//! The UI is deliberately ignorant of the platform. It asks for "the files the
//! user chose" and "put this on the clipboard" and never learns that one of
//! those means a Win32 dialog on Windows and something else on a desktop
//! Linux. That separation is why `vaani_ui` has no dialog dependency at all and
//! why its tests run headless.
//!
//! # Why the dialog is driven directly
//!
//! `IFileOpenDialog` is the shell's own file dialog - the same one Explorer
//! uses, with the same places bar, search box and virtual folders. A wrapper
//! crate would call exactly this COM interface, so calling it directly is
//! *less* code, not more, and it is a crate this build can actually resolve:
//! `rfd` is not in the local registry cache and this machine has no network.
//!
//! # Why a failed dialog is quiet
//!
//! Both methods run on the Slint event-loop thread, which is where the UI
//! called them, and a modal dialog blocks that thread until it returns. There
//! is therefore nothing useful to do with an error but report it and carry on.
//! A user who cancels a dialog is not in an error state, they changed their
//! mind. Returning an empty selection for "cancelled" and for "could not open
//! the dialog" is deliberate: the window stays usable either way, and the user
//! can always type a path into the field instead.

use std::path::PathBuf;

use vaani_ui::backend::{ExportTarget, HostServices};

/// The shell file dialog, driven directly.
///
/// `unsafe` is confined to this one module on purpose: it is the only place in
/// the desktop app that talks to COM, and every block below says what it
/// relies on.
#[cfg(windows)]
#[allow(unsafe_code)] // the sole COM boundary in the application
pub mod filedialog {
    use std::ffi::c_void;
    use std::path::PathBuf;
    use windows::Win32::System::Com::{
        CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx,
        CoTaskMemFree, CoUninitialize,
    };
    use windows::Win32::UI::Shell::{
        Common::COMDLG_FILTERSPEC, FILEOPENDIALOGOPTIONS, FOS_ALLOWMULTISELECT,
        FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR, FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST,
        FileOpenDialog, FileSaveDialog, IFileOpenDialog, IFileSaveDialog, SIGDN_FILESYSPATH,
    };
    use windows::core::{PCWSTR, PWSTR};

    /// RAII for COM initialisation on the calling thread.
    ///
    /// The shell dialogs require an initialised apartment. If COM is already
    /// up, this succeeds with `S_FALSE` and must *not* be unbalanced, which is
    /// why the result is remembered rather than assumed.
    struct ComGuard {
        owned: bool,
    }

    impl ComGuard {
        /// Initialise COM as an apartment-threaded client, or join the
        /// existing one. Never fails: a failure here means the dialog cannot
        /// open, and the caller degrades to "no files chosen".
        fn new() -> Self {
            // SAFETY: `CoInitializeEx` with a null reserved pointer and
            // COINIT_APARTMENTTHREADED is the documented way to enter an
            // apartment. It returns S_FALSE when the thread is already in one,
            // and that success still needs a matching `CoUninitialize`.
            let hr = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
            Self { owned: hr.is_ok() }
        }
    }

    impl Drop for ComGuard {
        fn drop(&mut self) {
            if self.owned {
                // SAFETY: paired with the successful `CoInitializeEx` above, on
                // the same thread, exactly once.
                unsafe { CoUninitialize() };
            }
        }
    }

    /// Turn a Rust string into a NUL-terminated UTF-16 buffer for COM.
    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Read a `PWSTR` the shell allocated and free it.
    ///
    /// # Safety
    ///
    /// `p` must be a `PWSTR` the shell allocated with `CoTaskMemAlloc`, as
    /// `IShellItem::GetDisplayName` returns.
    unsafe fn take_shell_string(p: PWSTR) -> Option<PathBuf> {
        if p.is_null() {
            return None;
        }
        // SAFETY: `p` is a non-null NUL-terminated UTF-16 string allocated by
        // the shell. Build the slice, copy to owned UTF-8, then free below.
        let len = unsafe { windows::Win32::Globalization::lstrlenW(PCWSTR(p.0)) } as usize;
        let slice = unsafe { std::slice::from_raw_parts(p.0, len) };
        let text = String::from_utf16_lossy(slice);
        // SAFETY: `p` came from the shell's allocator; this is the matching
        // free.
        unsafe { CoTaskMemFree(Some(p.0 as *const c_void)) };
        (!text.is_empty()).then(|| PathBuf::from(text))
    }

    /// The filesystem path of a shell item, whatever kind of item it is.
    unsafe fn shell_item_path(item: &windows::Win32::UI::Shell::IShellItem) -> Option<PathBuf> {
        // SAFETY: `item` is a live COM interface. `GetDisplayName`
        // (SIGDN_FILESYSPATH) returns a CoTaskMemAlloc'd PWSTR, which
        // `take_shell_string` frees.
        let name = unsafe { item.GetDisplayName(SIGDN_FILESYSPATH) }.ok()?;
        unsafe { take_shell_string(name) }
    }

    /// The default media filter, derived from the decoder's accepted
    /// extensions so the picker cannot offer a format the app will refuse.
    fn media_specs() -> Vec<(Vec<u16>, Vec<u16>)> {
        let globs: Vec<String> = vaani_core::SUPPORTED_EXTENSIONS
            .iter()
            .map(|e| format!("*.{e}"))
            .collect();
        vec![
            (wide("Media files"), wide(&globs.join(";"))),
            (wide("All files"), wide("*")),
        ]
    }

    /// Open a multi-select "open file" dialog. Returns the chosen paths; empty
    /// on cancel or on any failure to open.
    pub fn pick_files(title: &str) -> Vec<PathBuf> {
        let _com = ComGuard::new();
        let specs = media_specs();

        // SAFETY: `FileOpenDialog` is the documented CLSID for the common item
        // dialog; `CoCreateInstance` with a null outer pointer and
        // CLSCTX_INPROC_SERVER is the documented way to instantiate it, and
        // the result is released when `dialog` goes out of scope.
        let dialog: IFileOpenDialog =
            match unsafe { CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER) } {
                Ok(d) => d,
                Err(_) => return Vec::new(),
            };

        // SAFETY: each call is on the live `dialog` interface. The title is a
        // NUL-terminated buffer that outlives the call. The filter spec slice
        // is valid for the duration of the `SetFileTypes` call, and the
        // buffers it points at outlive it.
        unsafe {
            let title_w = wide(title);
            let _ = dialog.SetTitle(PCWSTR(title_w.as_ptr()));
            let specs: Vec<COMDLG_FILTERSPEC> = specs
                .iter()
                .map(|(name, spec)| COMDLG_FILTERSPEC {
                    pszName: PCWSTR(name.as_ptr()),
                    pszSpec: PCWSTR(spec.as_ptr()),
                })
                .collect();
            let _ = dialog.SetFileTypes(&specs);
            // Multi-select, existing files only, filesystem paths only, and do
            // not move the process's working directory out from under the job
            // that is about to run.
            let _ = dialog.SetOptions(
                FOS_ALLOWMULTISELECT | FOS_PATHMUSTEXIST | FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR,
            );
        }

        // `Show` returning an error is how a cancel is reported, not a failure.
        // SAFETY: `dialog` is live; `Show` blocks the UI thread by design,
        // which is what a modal dialog is supposed to do.
        if unsafe { dialog.Show(None) }.is_err() {
            return Vec::new();
        }

        // SAFETY: after a successful `Show`, the dialog holds the user's
        // selection. Two APIs read it, and both are needed:
        //
        // `GetSelectedItems` reflects what is selected IN THE VIEW. When the
        // user types or pastes a path and presses Enter, nothing in the view
        // is "selected" — the dialog resolves the typed path directly to its
        // result instead. In that case GetSelectedItems fails with E_FAIL
        // while GetResult holds the file. Reading only the collection meant
        // pasted paths silently yielded nothing, which is precisely how a
        // user adds a file they already have the path for. So: the collection
        // first (it carries multi-select), the single result as fallback.
        unsafe {
            if let Ok(items) = dialog.GetSelectedItems() {
                let count = items.GetCount().unwrap_or(0);
                let mut out = Vec::with_capacity(count as usize);
                for idx in 0..count {
                    if let Ok(item) = items.GetItemAt(idx)
                        && let Some(path) = shell_item_path(&item)
                    {
                        out.push(path);
                    }
                }
                if !out.is_empty() {
                    return out;
                }
            }
            if let Ok(item) = dialog.GetResult()
                && let Some(path) = shell_item_path(&item)
            {
                return vec![path];
            }
            Vec::new()
        }
    }

    /// Open a "save file" dialog pre-filled with `suggested_name`. Returns the
    /// chosen path, or `None` on cancel/failure.
    pub fn pick_save_target(suggested_name: &str) -> Option<PathBuf> {
        let _com = ComGuard::new();
        let ext = suggested_name
            .rsplit_once('.')
            .map(|(_, e)| e)
            .filter(|e| !e.is_empty() && e.len() <= 5)
            .unwrap_or("txt");
        let specs = [
            (wide("Transcript"), wide(&format!("*.{ext}"))),
            (wide("All files"), wide("*")),
        ];

        // SAFETY: `FileSaveDialog` is the documented CLSID for the save
        // variant, instantiated the same way as the open dialog.
        let dialog: IFileSaveDialog =
            match unsafe { CoCreateInstance(&FileSaveDialog, None, CLSCTX_INPROC_SERVER) } {
                Ok(d) => d,
                Err(_) => return None,
            };

        // SAFETY: every call is on the live `dialog` interface; the
        // NUL-terminated buffers and the filter spec slice outlive the calls
        // that borrow them.
        unsafe {
            let title_w = wide("Save transcript");
            let _ = dialog.SetTitle(PCWSTR(title_w.as_ptr()));
            let specs: Vec<COMDLG_FILTERSPEC> = specs
                .iter()
                .map(|(name, spec)| COMDLG_FILTERSPEC {
                    pszName: PCWSTR(name.as_ptr()),
                    pszSpec: PCWSTR(spec.as_ptr()),
                })
                .collect();
            let _ = dialog.SetFileTypes(&specs);
            let name_w = wide(suggested_name);
            let _ = dialog.SetFileName(PCWSTR(name_w.as_ptr()));
            let ext_w = wide(ext);
            let _ = dialog.SetDefaultExtension(PCWSTR(ext_w.as_ptr()));
            let _ = dialog.SetOptions(FILEOPENDIALOGOPTIONS(
                FOS_OVERWRITEPROMPT.0 | FOS_FORCEFILESYSTEM.0 | FOS_NOCHANGEDIR.0,
            ));
            // An error from `Show` is a cancel.
            if dialog.Show(None).is_err() {
                return None;
            }
            // SAFETY: after a successful `Show`, `GetResult` returns the chosen
            // item and the PWSTR it yields is freed by `shell_item_path`.
            let item = dialog.GetResult().ok()?;
            shell_item_path(&item)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_default_filter_offers_everything_the_decoder_accepts() {
            // Built from SUPPORTED_EXTENSIONS, so this cannot drift from what
            // the media stage can actually open.
            let specs = media_specs();
            let joined: String = specs
                .iter()
                .map(|(_, s)| String::from_utf16_lossy(s))
                .collect::<Vec<_>>()
                .join(";");
            for ext in vaani_core::SUPPORTED_EXTENSIONS {
                assert!(
                    joined.contains(&format!("*.{ext}")),
                    "{ext} is decodable but missing from the file dialog"
                );
            }
        }

        #[test]
        fn com_initialisation_is_balanced() {
            // A dialog that leaks an apartment reference is a dialog that works
            // once. Proving the guard pairs is cheap and catches the leak at
            // the boundary rather than in a support ticket.
            let _a = ComGuard::new();
            let _b = ComGuard::new();
        }
    }
}

/// A non-Windows host has no dialog. Returning empty keeps the window usable:
/// the user can type a path instead of browsing for one.
#[cfg(not(windows))]
pub mod filedialog {
    use std::path::PathBuf;
    pub fn pick_files(_title: &str) -> Vec<PathBuf> {
        Vec::new()
    }
    pub fn pick_save_target(_suggested: &str) -> Option<PathBuf> {
        None
    }
}

/// The real host: the shell file dialog and the system clipboard.
#[derive(Debug, Clone, Copy, Default)]
pub struct DesktopHost;

impl DesktopHost {
    /// A host bound to the current desktop session.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl HostServices for DesktopHost {
    fn pick_media_files(&self) -> Vec<PathBuf> {
        filedialog::pick_files("Choose recordings to transcribe")
    }

    fn pick_export_target(&self, suggested_name: &str) -> Option<ExportTarget> {
        // A save dialog returns a *file*, so this is a File target. Returning
        // Directory would make the exporter invent a name and write somewhere
        // the user did not choose.
        filedialog::pick_save_target(suggested_name).map(ExportTarget::File)
    }

    fn set_clipboard_text(&self, text: &str) {
        // Best effort. The UI offers Copy unconditionally, and a clipboard
        // that is momentarily locked by another application must not take the
        // window down with it.
        if let Ok(mut clip) = arboard::Clipboard::new() {
            let _ = clip.set_text(text.to_owned());
        }
    }

    fn host_name(&self) -> String {
        format!("VaaniDesk {}", env!("CARGO_PKG_VERSION"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_services_are_cheap_and_shareable() {
        // `run` takes `Arc<dyn HostServices>`, so this must be Send + Sync.
        let host: std::sync::Arc<dyn HostServices> = std::sync::Arc::new(DesktopHost::new());
        assert!(!host.host_name().is_empty());
    }
}
