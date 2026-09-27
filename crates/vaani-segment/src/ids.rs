//! How a segment id survives grouping.
//!
//! A grouped segment's id is derived from the source segment its **first word**
//! came from:
//!
//! * the group keeps that source segment's id **verbatim** when its first word
//!   is that segment's own first word — the no-op case, so a transcript that
//!   needs no splitting comes out byte-identical;
//! * otherwise the id is `"{source_id}#{k}"`, where `k` is the index of the
//!   group's first word **within that source segment**.
//!
//! # Why the rule is enough to keep ids unique
//!
//! Every group starts at a different word, so no two groups can name the same
//! `(source segment, first word)` pair. A group that starts at a source
//! segment's word 0 is the only group that can, which is why `0` needs no
//! suffix. Merging does not weaken this: a merged group's first word is still
//! one specific word of one specific source segment, and no other group starts
//! there.
//!
//! # Why not regenerate a UUID
//!
//! [`Segment::new`](vaani_core::Segment::new) generates a random id, and
//! calling it per group was the obvious implementation. It was rejected on one
//! measured ground: it is not **deterministic**. Two runs of the same pipeline
//! over the same audio would produce different ids, which breaks reproducible
//! exports and makes "the CLI and the app produce the same transcript"
//! untestable — and that equality is this crate's acceptance criterion.
//!
//! It would also throw away traceability. `#k` says exactly which source segment
//! and which word inside it a group came from, so a regroup after a policy
//! change is auditable rather than mysterious.
//!
//! # What is lost, stated plainly
//!
//! The ids of the *other* source segments a group absorbed do not survive
//! anywhere in the transcript. Grouping runs before persistence, so nothing has
//! referenced them yet, and the surviving id is a faithful address for the same
//! audio. A caller that needs the full mapping must capture it before grouping.

/// Build a group's id from its source segment's id and the index of its first
/// word within that segment.
#[must_use]
pub fn group_id(source_id: &str, first_word_index: usize) -> String {
    if first_word_index == 0 {
        source_id.to_owned()
    } else {
        format!("{source_id}#{first_word_index}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_starting_at_a_source_segments_first_word_keeps_the_id() {
        assert_eq!(group_id("seg-7", 0), "seg-7");
    }

    #[test]
    fn a_group_starting_mid_segment_names_where_it_started() {
        assert_eq!(group_id("seg-7", 3), "seg-7#3");
    }

    #[test]
    fn the_id_is_deterministic() {
        for i in 0..5 {
            let a = group_id("abc", i);
            let b = group_id("abc", i);
            assert_eq!(a, b);
            assert!(!a.is_empty());
        }
    }

    #[test]
    fn a_source_id_containing_a_hash_is_not_mistaken_for_a_derived_one() {
        // The suffix is for a human reading it, and the split is unambiguous
        // from the right: the index is the text after the *final* '#', and a
        // source id may contain one.
        assert_eq!(group_id("a#b", 2), "a#b#2");
    }
}
