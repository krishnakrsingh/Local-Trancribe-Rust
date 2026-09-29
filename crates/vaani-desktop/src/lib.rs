//! The VaaniDesk desktop application, as a library.
//!
//! The binary in `main.rs` is intentionally thin: it opens the store, builds
//! the backend, and runs the window. Everything it needs to do that — the
//! [`host::DesktopHost`] with the real file dialog and clipboard — lives here
//! so it can be tested without launching a window.
//!
//! The file dialog is the riskiest code in the product that no test had ever
//! touched: COM initialisation, interface casts, string marshalling, and a
//! modal loop, all written by hand. `dialog_probe` below opens it on a thread,
//! finds its window, dismisses it, and asserts the return — with no user
//! input, so a broken dialog fails the suite instead of hanging a release.

pub mod host;
