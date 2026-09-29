//! Proves the native file dialog opens, displays, and cancels — with no user input.
//!
//! The dialog is the riskiest untested code in the product: hand-written COM,
//! string marshalling, and a modal loop. This spawns `pick_files` on a thread,
//! polls for a dialog window, closes it with `WM_CLOSE` (what ESC produces),
//! and asserts the call returns empty. A broken dialog fails here instead of
//! hanging a release.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use vaani_desktop::host::filedialog;
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextLengthW, GetWindowTextW, PostMessageW, WM_CLOSE,
};

/// Every top-level window title, for finding the dialog.
fn top_level_titles() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    unsafe extern "system" fn visit(hwnd: HWND, state: LPARAM) -> windows::core::BOOL {
        // SAFETY: `state` is `&mut Vec<String>` as an integer, put there by
        // the caller below. It must be the address OF the Vec, not of its
        // elements: `vec.as_mut_ptr()` points at the heap buffer, and
        // interpreting that as the 24-byte Vec struct corrupts the heap.
        // That exact mistake AV'd this probe before it was found.
        let titles = unsafe { &mut *(state.0 as *mut Vec<String>) };
        // SAFETY: `hwnd` is a live top-level window supplied by the enumerator.
        let len = unsafe { GetWindowTextLengthW(hwnd) };
        if len > 0 {
            let mut buf = vec![0u16; (len + 1) as usize];
            let read = unsafe { GetWindowTextW(hwnd, &mut buf) };
            if read > 0 {
                titles.push(String::from_utf16_lossy(&buf[..read as usize]));
            }
        }
        true.into()
    }
    // SAFETY: `visit` only appends to `out` and returns TRUE to continue.
    // `&mut out` is the Vec struct itself; see the note above about why
    // `out.as_mut_ptr()` would be wrong here.
    unsafe {
        let _ = EnumWindows(Some(visit), LPARAM(&mut out as *mut Vec<String> as isize));
    }
    out
}

#[test]
fn the_file_dialog_opens_and_cancel_returns_empty() {
    eprintln!("probe: spawning dialog thread");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        eprintln!("probe dialog thread: calling pick_files");
        let files = filedialog::pick_files("VaaniDesk probe dialog");
        eprintln!(
            "probe dialog thread: pick_files returned {} files",
            files.len()
        );
        tx.send(files).expect("send");
    });

    // A dialog that takes longer than this to appear is broken for a user too.
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut closed = false;
    let mut polls = 0;
    while Instant::now() < deadline {
        polls += 1;
        if polls == 1 {
            eprintln!("probe main: about to enumerate windows");
        }
        // Find the probe dialog by its title and close it the way ESC does.
        let titles = top_level_titles();
        if polls == 1 {
            eprintln!("probe main: first poll saw {} windows", titles.len());
        }
        let found = titles
            .into_iter()
            .any(|t| t.contains("VaaniDesk probe dialog"));
        if found {
            // Close every candidate window titled like the probe. WM_CLOSE is
            // what the dialog's own cancel path produces.
            unsafe extern "system" fn close_matching(
                hwnd: HWND,
                state: LPARAM,
            ) -> windows::core::BOOL {
                let needle = unsafe { &*(state.0 as *const String) };
                let len = unsafe { GetWindowTextLengthW(hwnd) };
                if len > 0 {
                    let mut buf = vec![0u16; (len + 1) as usize];
                    let read = unsafe { GetWindowTextW(hwnd, &mut buf) };
                    if read > 0 {
                        let title = String::from_utf16_lossy(&buf[..read as usize]);
                        if title.contains(needle.as_str()) {
                            // SAFETY: closing a window we do not own is what a
                            // user pressing ESC does; the dialog handles it.
                            unsafe {
                                let _ = PostMessageW(Some(hwnd), WM_CLOSE, WPARAM(0), LPARAM(0));
                            }
                        }
                    }
                }
                true.into()
            }
            let needle = "VaaniDesk probe dialog".to_string();
            unsafe {
                let _ = EnumWindows(
                    Some(close_matching),
                    LPARAM(&needle as *const String as isize),
                );
            }
            closed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(closed, "no dialog window appeared within 15 s");

    match rx.recv_timeout(Duration::from_secs(15)) {
        Ok(files) => assert!(
            files.is_empty(),
            "closing the dialog should yield an empty selection, got {files:?}"
        ),
        Err(_) => panic!("the dialog call did not return after its window was closed"),
    }
}
