//! The sys probes run on the UI startup path's cousin (the bench startup
//! path): whatever the machine looks like, they must return a coherent answer
//! rather than panic.

use vaani_bench::sys;

#[test]
fn rss_probe_does_not_panic_and_is_self_consistent() {
    let bytes = sys::peak_rss_bytes();
    let mb = sys::peak_rss_mb();
    assert!(mb >= 0.0);
    match bytes {
        // Two successive probes: RSS can move between calls (the allocator
        // is live), so the invariant is positivity + agreement on
        // measured-vs-unmeasured, not exact equality.
        Some(b) => {
            assert!(b > 0, "a live test process has a nonzero working set");
            assert!(mb > 0.0);
        }
        None => assert_eq!(mb, 0.0, "unmeasured RSS must be the 0.0 marker"),
    }
}

#[test]
fn adapter_enumeration_does_not_panic() {
    // Any machine shape is legitimate (including zero adapters on headless
    // CI); the contract is "no panic, coherent struct".
    let _ = sys::adapters();
    let _ = sys::inference_vram_budget_bytes();
    let _ = sys::peak_vram_mb();
    assert!(!sys::device_label().is_empty());
    assert!(!sys::git_sha().is_empty());
}

#[test]
fn vram_budget_implies_a_real_inference_adapter() {
    // Guards the inference_adapter-vs-preferred_adapter distinction: a
    // reported budget must come from an NVIDIA non-software adapter while a
    // CUDA runtime loads, never from the largest (possibly iGPU) budget.
    if sys::inference_vram_budget_bytes().is_some() {
        assert!(sys::cudart_present());
        assert!(sys::peak_vram_mb().is_some());
        assert!(sys::device_label().starts_with("cuda"));
    } else {
        assert!(sys::peak_vram_mb().is_none());
        assert_eq!(sys::device_label(), "cpu");
    }
}
