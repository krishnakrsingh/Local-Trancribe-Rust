//! Compiles `ui/app-window.slint` into Rust code at build time.
//!
//! `cargo:rerun-if-changed` is declared for every `.slint` file so that editing
//! a property in the markup rebuilds only this crate.

fn main() {
    let ui = std::path::Path::new("ui");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", ui.display());
    for entry in walk(ui) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
    slint_build::compile_with_config(
        "ui/app-window.slint",
        slint_build::CompilerConfiguration::new().with_style("fluent".to_owned()),
    )
    .expect(
        "ui/app-window.slint must compile; a broken .slint file is a build error, not a warning",
    );
}

/// Every `.slint` file under `dir`, so the build script tracks the whole tree.
fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else if path.extension().is_some_and(|e| e == "slint") {
            out.push(path);
        }
    }
    out
}
