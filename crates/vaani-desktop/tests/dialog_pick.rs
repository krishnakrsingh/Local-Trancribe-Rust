//! Proves the dialog success path: a real file selection returns a real path.
//!
//! Opens `pick_files` on a thread, finds the dialog window, writes a known
//! file path directly into its filename field (`WM_SETTEXT` on the Edit
//! inside the filename combo), clicks Open (`BM_CLICK`), and asserts
//! `pick_files` returns exactly that path. This exercises everything cancel
//! does not: `GetSelectedItems`, `GetItemAt`, `GetDisplayName`,
//! `take_shell_string` (UTF-16 marshalling back), and `CoTaskMemFree`.
//!
//! If any of that is broken, "Add file" silently yields nothing and the user
//! can never transcribe — which is precisely the reported symptom.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use vaani_desktop::host::filedialog;
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    BM_CLICK, EnumChildWindows, EnumWindows, GetWindowTextLengthW, GetWindowTextW, SendMessageW,
    WM_SETTEXT,
};

fn window_text(hwnd: HWND) -> String {
    // SAFETY: hwnd is a live window supplied by the enumerator.
    let len = unsafe { GetWindowTextLengthW(hwnd) };
    if len <= 0 {
        return String::new();
    }
    let mut buf = vec![0u16; (len + 1) as usize];
    let read = unsafe { GetWindowTextW(hwnd, &mut buf) };
    if read <= 0 {
        return String::new();
    }
    String::from_utf16_lossy(&buf[..read as usize])
}

fn class_name(hwnd: HWND) -> String {
    let mut buf = vec![0u16; 64];
    // SAFETY: hwnd is live; the buffer is ours.
    let read = unsafe { windows::Win32::UI::WindowsAndMessaging::GetClassNameW(hwnd, &mut buf) };
    String::from_utf16_lossy(&buf[..read as usize])
}

fn find_top_window_titled(needle: &str) -> Option<HWND> {
    struct Found(Option<HWND>, String);
    unsafe extern "system" fn visit(hwnd: HWND, state: LPARAM) -> windows::core::BOOL {
        // SAFETY: state is &mut Found for the duration of this call.
        let found = unsafe { &mut *(state.0 as *mut Found) };
        if window_text(hwnd).contains(&found.1) {
            found.0 = Some(hwnd);
        }
        true.into()
    }
    let mut found = Found(None, needle.to_string());
    // SAFETY: visit only writes `found` and returns TRUE.
    unsafe {
        let _ = EnumWindows(Some(visit), LPARAM(&mut found as *mut Found as isize));
    }
    found.0
}

/// The filename Edit inside `parent` (the filename ComboBoxEx32, or any Edit).
fn find_edit(parent: HWND, needle_class: &str) -> Option<HWND> {
    struct Found(Option<HWND>, String);
    unsafe extern "system" fn visit(hwnd: HWND, state: LPARAM) -> windows::core::BOOL {
        // SAFETY: state is &mut Found for the duration of this call.
        let found = unsafe { &mut *(state.0 as *mut Found) };
        if class_name(hwnd) == found.1 {
            found.0 = Some(hwnd);
            return false.into(); // stop at the first match
        }
        true.into()
    }
    let mut found = Found(None, needle_class.to_string());
    // SAFETY: visit only writes `found`.
    unsafe {
        let _ = EnumChildWindows(
            Some(parent),
            Some(visit),
            LPARAM(&mut found as *mut Found as isize),
        );
    }
    found.0
}

/// A direct child button whose text matches, e.g. the Open button.
fn find_button(parent: HWND, label_substring: &str) -> Option<HWND> {
    struct Found(Option<HWND>, String);
    unsafe extern "system" fn visit(hwnd: HWND, state: LPARAM) -> windows::core::BOOL {
        // SAFETY: state is &mut Found for the duration of this call.
        let found = unsafe { &mut *(state.0 as *mut Found) };
        if class_name(hwnd) == "Button" && window_text(hwnd).contains(&found.1) {
            found.0 = Some(hwnd);
            return false.into();
        }
        true.into()
    }
    let mut found = Found(None, label_substring.to_string());
    unsafe {
        let _ = EnumChildWindows(
            Some(parent),
            Some(visit),
            LPARAM(&mut found as *mut Found as isize),
        );
    }
    found.0
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[test]
#[ignore = "drives the real dialog; needs a desktop session, takes ~10 s"]
fn picking_a_file_returns_its_path() {
    // A real media file: the dialog filters to media types, and picking a
    // file outside the filter (like Cargo.toml) is a different question than
    // picking one inside it. Selection does not read the file, so its 50 MB
    // size is irrelevant. Skips cleanly when absent.
    //
    // Canonicalized: the raw join contains `..` segments, and the shell fails
    // GetSelectedItems with E_FAIL on the un-normalized path even though Show
    // accepts it. A real dialog pick never contains `..` (the shell returns
    // clean paths), so the test must too — otherwise it tests a path shape no
    // user can produce.
    let raw = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("scratch")
        .join("clip.wav");
    let Ok(canonical) = raw.canonicalize() else {
        println!("SKIP: scratch/clip.wav is absent");
        return;
    };
    // canonicalize() yields a `\\?\`-prefixed verbatim path, which the file
    // dialog cannot resolve (it hangs waiting on an error the test cannot
    // see). Strip the prefix: the result is still absolute and canonical.
    let target = PathBuf::from(
        canonical
            .display()
            .to_string()
            .trim_start_matches(r"\\?\")
            .to_owned(),
    );
    let target_str = target.display().to_string();

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        tx.send(filedialog::pick_files("VaaniDesk pick probe"))
            .expect("send");
    });

    // Wait for the dialog window.
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut dialog = None;
    while Instant::now() < deadline {
        dialog = find_top_window_titled("VaaniDesk pick probe");
        if dialog.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let dialog = dialog.expect("the dialog must appear");
    // Let it finish constructing before driving its children.
    std::thread::sleep(Duration::from_millis(800));

    // Write the path into the filename field. The common item dialog hosts it
    // as ComboBoxEx32 > ComboBox > Edit; the direct Edit child of the
    // ComboBoxEx32 is a different control and setting it does nothing the
    // dialog will honour. Driving the wrong Edit was a real failure this test
    // caught: the dialog opened, Open was clicked, and the selection came
    // back empty.
    let combo_ex = find_edit(dialog, "ComboBoxEx32").expect("filename combo must exist");
    let combo = find_edit(combo_ex, "ComboBox").expect("inner combo must exist");
    let edit = find_edit(combo, "Edit").expect("filename edit must exist");
    let path_w = wide(&target_str);
    // SAFETY: edit is a live Edit control; WM_SETTEXT with a NUL-terminated
    // buffer we own is its documented input path.
    unsafe {
        SendMessageW(
            edit,
            WM_SETTEXT,
            Some(WPARAM(0)),
            Some(LPARAM(path_w.as_ptr() as isize)),
        );
    }
    // Read it back. If the text is not there, the Edit handle is wrong and
    // everything downstream is meaningless — fail here with the actual state
    // rather than debugging an empty selection later.
    let back = window_text(edit);
    assert_eq!(
        back, target_str,
        "the filename field must hold the path we set (wrong Edit control?)"
    );
    std::thread::sleep(Duration::from_millis(300));

    // Drive the dialog the way a user does: click the filename field with the
    // mouse, type the path, press Enter. Every earlier approach failed for a
    // faithful reason:
    //   * WM_SETTEXT sets text without EN_CHANGE, so the dialog never parses it.
    //   * EN_CHANGE synthesized by hand plus BM_CLICK closed the dialog with an
    //     empty selection (Show Ok, GetSelectedItems E_FAIL).
    //   * WM_KEYDOWN/Enter direct to the Edit dies in its proc (no dialog loop).
    // SendInput goes through the real input queue: focus, change
    // notifications, validation and accept all happen exactly as for typing.
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_KEYUP,
        MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE,
        MOUSEINPUT, SendInput, VIRTUAL_KEY,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};

    fn send(inputs: &[INPUT]) {
        // SAFETY: well-formed inputs; SendInput copies them before returning.
        unsafe {
            SendInput(inputs, std::mem::size_of::<INPUT>() as i32);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    // Screen size for absolute mouse coordinates (0..65535 across the screen).
    let (cx, cy) = unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) };
    // Click the middle of the filename field to focus it.
    let rect = {
        let mut r = windows::Win32::Foundation::RECT::default();
        // SAFETY: edit is a live control; GetWindowRect fills our RECT.
        unsafe {
            let _ = windows::Win32::UI::WindowsAndMessaging::GetWindowRect(edit, &mut r);
        }
        r
    };
    let mx = ((rect.left + rect.right) / 2).max(0);
    let my = ((rect.top + rect.bottom) / 2).max(0);
    let ax = (mx * 65535 / cx.max(1)) as i32;
    let ay = (my * 65535 / cy.max(1)) as i32;
    let mouse = |flags, dx: i32, dy: i32| INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx,
                dy,
                mouseData: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    send(&[mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)]);
    send(&[mouse(MOUSEEVENTF_LEFTDOWN, 0, 0)]);
    send(&[mouse(MOUSEEVENTF_LEFTUP, 0, 0)]);
    std::thread::sleep(Duration::from_millis(300));

    fn key(vk: VIRTUAL_KEY, up: bool) {
        let input = INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: if up {
                        KEYEVENTF_KEYUP
                    } else {
                        Default::default()
                    },
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        send(&[input]);
    }
    // Paste the path from the clipboard instead of typing it key by key.
    // Typing an 80-character path at 10 ms a key is 170 synthetic inputs;
    // the first few are easily lost before focus settles, and a truncated
    // path fails validation downstream with no useful error. Pasting is three
    // inputs, goes through the real clipboard and the real EN_CHANGE path,
    // and is what users actually do with long paths.
    {
        use windows::Win32::UI::Input::KeyboardAndMouse::{VK_CONTROL, VK_SHIFT};
        let mut clip = arboard::Clipboard::new().expect("clipboard must exist for paste");
        clip.set_text(target_str.clone()).expect("clipboard set");
        // Settle: the click above must have taken effect before keys arrive.
        std::thread::sleep(Duration::from_millis(800));
        // Ctrl+A to replace any placeholder, then Ctrl+V to paste.
        let modified = |vk, ctrl_down: bool, shift_down: bool| {
            if ctrl_down {
                key(VK_CONTROL, false);
            }
            if shift_down {
                key(VK_SHIFT, false);
            }
            key(vk, false);
            key(vk, true);
            if shift_down {
                key(VK_SHIFT, true);
            }
            if ctrl_down {
                key(VK_CONTROL, true);
            }
        };
        modified(VIRTUAL_KEY(0x41), true, false); // Ctrl+A
        std::thread::sleep(Duration::from_millis(200));
        modified(VIRTUAL_KEY(0x56), true, false); // Ctrl+V
        std::thread::sleep(Duration::from_millis(500));
    }
    // Read back what actually landed. If this does not match, input went
    // elsewhere and everything after is meaningless — fail here with both
    // strings rather than debugging an empty selection later.
    let landed = window_text(edit);
    assert_eq!(
        landed, target_str,
        "pasted text must land in the filename field"
    );
    // Click Open deterministically. Paste went through the real input queue so
    // EN_CHANGE fired and the dialog parsed the path; BM_CLICK via SendMessage
    // needs no focus and cannot miss, unlike an Enter keystroke that must land
    // on the right window at the right moment.
    let open = find_button(dialog, "Open").expect("the dialog must have an Open button");
    // SAFETY: open is a live Button; BM_CLICK is a synthesized click.
    unsafe {
        SendMessageW(open, BM_CLICK, Some(WPARAM(0)), Some(LPARAM(0)));
    }

    match rx.recv_timeout(Duration::from_secs(20)) {
        Ok(files) => assert_eq!(
            files,
            vec![target],
            "picking a file must return exactly its path"
        ),
        Err(_) => panic!("pick_files did not return after Open was clicked"),
    }
}
