//! The conformance gate: `run_all_conformance()` must be fully green.
//!
//! Every check has a name; a failure names the contract rule that broke. This
//! file asserts the harness itself stays honest: no check may fail on the real
//! crates with synthetic audio.

use vaani_qa::conformance::run_all_conformance;

#[test]
fn all_conformance_checks_pass_on_the_real_crates() {
    let report = run_all_conformance();
    for outcome in &report.outcomes {
        println!(
            "{} {} — {}",
            if outcome.passed { "PASS" } else { "FAIL" },
            outcome.name,
            outcome.detail
        );
    }
    assert!(
        report.ok(),
        "{}/{} conformance checks failed",
        report.failed(),
        report.outcomes.len()
    );
    assert!(
        report.outcomes.len() >= 20,
        "the harness must pin at least 20 checks, found {}",
        report.outcomes.len()
    );
}
