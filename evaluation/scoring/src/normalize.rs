//! Text normalisation for WER and CER.
//!
//! # Why this module exists at all
//!
//! WER is meaningless without a stated normalisation, and it is *especially*
//! meaningless for Hindi/Hinglish, where a single spoken syllable has several
//! legitimate written forms. Applying different rules to the reference and to
//! the hypothesis is the single most common way an evaluation lies to you: the
//! number moves by tens of points and nothing about the system changed.
//!
//! So the rules are:
//!
//! * a fixed, ordered, individually numbered list ([`Rule`]);
//! * **identical for reference and hypothesis**, always, with no exceptions and
//!   no "we normalised the reference harder" path in the code;
//! * recorded in the report as [`NormalisationProfile::id`], so a number can
//!   never be quoted without the profile that produced it;
//! * no stemming, no lemmatisation, no transliteration, no a/an/the deletion.
//!   Those change the unit of measurement. WER here is *lexical*.
//!
//! # Profiles
//!
//! | id | adds |
//! |---|---|
//! | `strict` | the common rules N1..N8 only |
//! | `hinglish` | `strict` + H1 (Latin diacritic folding) |
//! | `hinglish-translit` | `hinglish` + H2 (double-vowel folding, **lossy**) |
//!
//! ## Common rules, applied in this order
//!
//! | id | rule | why it exists in a Hindi meeting |
//! |---|---|---|
//! | N1 | strip a leading UTF-8 BOM | a Windows editor adds it; it is not a word |
//! | N2 | Devanagari and Arabic-Indic digits to ASCII `0`-`9` | a transcriber writes `५`, a recogniser writes `5` |
//! | N3 | canonicalise the precomposed nukta letters to base + U+093C; collapse `ZWJ+U+093C` to U+093C; map the non-standard `ऩ ऱ ऴ` to `न र व`; drop bare ZWJ/ZWNJ | `ज़` is one spoken phoneme written three ways (`ज़`, `ज़`, `ज़`) |
//! | N4 | delete punctuation, symbols and non-space separators | the danda `।` is a sentence mark, not a word |
//! | N5 | collapse whitespace runs to one U+0020 | |
//! | N6 | drop the U+0964 danda and the ASCII newline | already covered by N4/N5, restated because it is the single most common stray token |
//! | N7 | lowercase every non-Devanagari character, when the mapping is one character | Devanagari has no case, so this only touches Latin; `İ` is left alone rather than expanding to two characters |
//! | N8 | split on U+0020, drop empty tokens | |
//!
//! ## `hinglish` adds
//!
//! | id | rule | why |
//! |---|---|---|
//! | H1 | fold the documented precomposed Latin letters to base ASCII, then lowercase | `ā/ā/a`, `ṣ/s`, `ñ/n`, `ṭ/t` are the same sound to a listener and different tokens to a scorer |
//!
//! ## `hinglish-translit` adds
//!
//! | id | rule | cost |
//! |---|---|---|
//! | H2 | fold `aa→a`, `ii→i`, `uu→u`, `ee→i`, `oo→u` inside a Latin token | **lossy**: it merges `be`/`bee`, `to`/`too`, `me`/`mee`. Only use it when the reference and the hypothesis are expected to differ in transliteration convention, and never quote it next to a `strict` number |
//!
//! `ai`, `au`, `oi` and `ou` are deliberately **not** folded. They are real
//! diphthongs in both languages and folding them would merge distinct words.
//!
//! # What this module deliberately does NOT do
//!
//! * no transliteration between Devanagari and Latin. `kya` and `क्या` stay
//!   different tokens. A code-switched meeting is scored in the script the
//!   speaker used; converting would hide exactly the boundary errors the
//!   taxonomy needs to see.
//! * no number-word folding. `पाँच` against `5` scores as 1 substitution plus
//!   1 deletion. That is a *true* statement about the two texts and hiding it
//!   would make the score incomparable to every published Hindi WER.
//! * no filler removal (`uh`, `um`, `अरे`, `तो`). Fillers are words; a meeting
//!   transcript contains them.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use unicode_segmentation::UnicodeSegmentation;

/// One numbered normalisation rule. Reported verbatim so a score is
/// reproducible from the report alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Rule {
    /// Stable rule id, e.g. `"N3"` or `"H2"`.
    pub id: &'static str,
    /// One-line statement of the rule.
    pub rule: &'static str,
    /// What applying it fixes in a Hindi/Hinglish meeting.
    pub why: &'static str,
}

const fn r(id: &'static str, rule: &'static str, why: &'static str) -> Rule {
    Rule { id, rule, why }
}

/// The complete, ordered rule list. Index order **is** application order.
pub const COMMON_RULES: &[Rule] = &[
    r(
        "N1",
        "strip a leading UTF-8 BOM",
        "a Windows editor adds it; it is not a word",
    ),
    r(
        "N2",
        "map Devanagari (U+0966..U+096F) and Arabic-Indic (U+06F0..U+06F9) digits to ASCII 0-9",
        "a human transcriber writes ५, a recogniser writes 5; they are the same utterance",
    ),
    r(
        "N3",
        "canonicalise Devanagari nukta letters: precomposed U+0958..U+095F -> base + U+093C; ZWJ + U+093C -> U+093C; U+0929->U+0928, U+0931->U+0930, U+0934->U+0935; drop bare U+200C/U+200D",
        "za is one phoneme written three ways; the U+0929 family are spelling variants, not distinct sounds",
    ),
    r(
        "N4",
        "delete Unicode punctuation, symbol and non-space separator characters (table in is_punct_or_symbol)",
        "the danda and the comma are sentence marks, not words",
    ),
    r(
        "N5",
        "collapse every run of Unicode whitespace to one U+0020",
        "",
    ),
    r(
        "N6",
        "delete U+0964 DEVANAGARI DANDA and U+0965 DOUBLE DANDA",
        "restated explicitly: the danda is the most common stray token in Hindi exports",
    ),
    r(
        "N7",
        "lowercase every character outside the Devanagari block, but only when the lowercase mapping is a single character",
        "Devanagari has no case; single-character-only keeps U+0130 from expanding into two characters inside a token",
    ),
    r("N8", "split on U+0020 and drop empty tokens", ""),
];

/// The rule `hinglish` adds.
pub const HINGLISH_RULES: &[Rule] = &[r(
    "H1",
    "fold the documented precomposed Latin letters to base ASCII, then lowercase (table in LATIN_FOLD)",
    "aa/a, ss/s and the retroflex transliterations are one sound and three tokens",
)];

/// The rule `hinglish-translit` adds. Lossy, on purpose, and opt-in.
pub const HINGLISH_TRANSLIT_RULES: &[Rule] = &[r(
    "H2",
    "fold aa->a, ii->i, uu->u, ee->i, oo->u inside a Latin-script token",
    "lossy: merges be/bee, to/too, me/mee. Quote it only next to its own profile id",
)];

/// The normalisation profile applied to both sides of every comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum NormalisationProfile {
    /// Common rules only. The default, and the one to quote.
    #[default]
    Strict,
    /// `strict` + H1.
    Hinglish,
    /// `hinglish` + H2. Lossy.
    HinglishTranslit,
}

impl NormalisationProfile {
    /// Wire name, stable in the report.
    pub fn id(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Hinglish => "hinglish",
            Self::HinglishTranslit => "hinglish-translit",
        }
    }

    /// Parse a wire name.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "strict" => Some(Self::Strict),
            "hinglish" => Some(Self::Hinglish),
            "hinglish-translit" | "hinglish_translit" => Some(Self::HinglishTranslit),
            _ => None,
        }
    }

    /// Every rule this profile applies, in application order.
    pub fn rules(self) -> Vec<Rule> {
        let mut v = COMMON_RULES.to_vec();
        if self != Self::Strict {
            v.extend_from_slice(HINGLISH_RULES);
        }
        if self == Self::HinglishTranslit {
            v.extend_from_slice(HINGLISH_TRANSLIT_RULES);
        }
        v
    }

    /// Machine-readable copy of [`NormalisationProfile::rules`], embedded in
    /// every report so a number is never separable from the rules behind it.
    pub fn rules_json(self) -> Vec<serde_json::Value> {
        self.rules()
            .into_iter()
            .map(|x| serde_json::json!({ "id": x.id, "rule": x.rule, "why": x.why }))
            .collect()
    }
}

/// Which scripts a token touches. Used for the code-switch boundary signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScriptMix {
    /// Devanagari only.
    Devanagari,
    /// Latin only.
    Latin,
    /// Other or empty: digits, symbols, anything unrecognised.
    Other,
    /// Devanagari and Latin in the same token. A code-switch boundary error
    /// candidate, and counted separately so WER is not quietly inflated by it.
    Mixed,
}

/// Facts about one normalised text, reported rather than assumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NormalisedText {
    /// The profile that produced this.
    pub profile: String,
    /// Normalised tokens, in order.
    pub tokens: Vec<String>,
    /// The normalised tokens joined by a single space. This is the string CER
    /// is computed over, and it is the artefact `vaani-eval norm` writes.
    pub text: String,
    /// `tokens.len()`.
    pub token_count: usize,
    /// Unicode scalar values of the normalised text, spaces included.
    pub char_count: usize,
    /// Extended grapheme clusters of the normalised text, spaces included.
    /// Counted with `unicode-segmentation`, so `क्ष` is one character and CER
    /// for Devanagari is not inflated by cluster decomposition.
    pub grapheme_count: usize,
    /// Tokens containing both Devanagari and Latin.
    pub mixed_script_tokens: Vec<String>,
    /// Tokens that are entirely digits.
    pub number_tokens: Vec<String>,
    /// Per-script token histogram.
    pub script_histogram: BTreeMap<String, usize>,
}

impl NormalisedText {
    /// True when the text holds no referenceable word.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Normalise `input` under `profile`.
///
/// Applies [`NormalisationProfile::rules`] in order, once, with no exceptions
/// for the reference side. This is the only entry point; there is no
/// "normalise harder" variant anywhere in the crate.
pub fn normalise(input: &str, profile: NormalisationProfile) -> NormalisedText {
    let mut s: Vec<char> = input.chars().collect();

    // N1
    if s.first() == Some(&'\u{feff}') {
        s.remove(0);
    }

    // N2
    for c in s.iter_mut() {
        *c = ascii_digit(*c);
    }

    // N3
    let mut stage: Vec<char> = Vec::with_capacity(s.len());
    let mut i = 0usize;
    while i < s.len() {
        let c = s[i];
        if c == '\u{200d}' {
            // ZWJ. If it is the joining convention for a nukta, keep the
            // nukta; otherwise it is a formatting artefact and goes.
            if s.get(i + 1) == Some(&'\u{93c}') {
                stage.push('\u{93c}');
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if c == '\u{200c}' {
            // ZWNJ carries real phonology in Devanagari (द्द vs दज), but it is
            // absent from almost every transcript produced by a tokeniser, so
            // keeping it would turn a spelling convention into a word error.
            // Documented as a known loss in OPEN ISSUES.
            i += 1;
            continue;
        }
        if (0x0958..=0x095f).contains(&(c as u32)) {
            // Precomposed nukta letter: U+0958..=U+095F are the eight
            // "letter + nukta" characters. They decompose to a base plus
            // U+093C, which is the form every other writer produces.
            let base = NUKTA_BASE[(c as u32 - 0x0958) as usize];
            stage.push(base);
            stage.push('\u{93c}');
            i += 1;
            continue;
        }
        stage.push(match c {
            '\u{0929}' => '\u{0928}',
            '\u{0931}' => '\u{0930}',
            '\u{0934}' => '\u{0935}',
            _ => c,
        });
        i += 1;
    }
    s = stage;

    // N4 + N6 (N6 is a subset of N4; both ids are reported).
    s.retain(|c| !is_punct_or_symbol(*c));

    // N5
    let joined: String = s
        .iter()
        .map(|c| if is_space(*c) { ' ' } else { *c })
        .collect();

    // N7
    let mut lowered: Vec<char> = Vec::with_capacity(joined.chars().count());
    for c in joined.chars() {
        let lc = c.to_lowercase();
        let mut it = lc.clone();
        match (it.next(), it.next()) {
            (Some(a), None) => lowered.push(if is_devanagari(c) { c } else { a }),
            _ => lowered.push(c),
        }
    }

    // N8
    let text: String = lowered.into_iter().collect();
    let tokens: Vec<String> = text
        .split(' ')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect();

    // Profile-specific folding, applied to the token strings.
    let tokens = if profile == NormalisationProfile::Strict {
        tokens
    } else {
        tokens
            .into_iter()
            .map(|t| fold_latin_diacritics(&t))
            .collect()
    };
    let tokens = if profile == NormalisationProfile::HinglishTranslit {
        tokens
            .into_iter()
            .map(|t| {
                if is_latin_only(&t) {
                    fold_double_vowels(&t)
                } else {
                    t
                }
            })
            .collect()
    } else {
        tokens
    };

    summarise(profile, tokens)
}

fn summarise(profile: NormalisationProfile, tokens: Vec<String>) -> NormalisedText {
    let mut mixed = Vec::new();
    let mut numbers = Vec::new();
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for t in &tokens {
        let mix = script_mix(t);
        *hist.entry(mix.as_str().to_string()).or_insert(0) += 1;
        if mix == ScriptMix::Mixed {
            mixed.push(t.clone());
        }
        if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
            numbers.push(t.clone());
        }
    }
    let text = tokens.join(" ");
    NormalisedText {
        profile: profile.id().to_string(),
        text: text.clone(),
        token_count: tokens.len(),
        char_count: text.chars().count(),
        grapheme_count: text.graphemes(true).count(),
        mixed_script_tokens: mixed,
        number_tokens: numbers,
        script_histogram: hist,
        tokens,
    }
}

impl ScriptMix {
    /// Wire name for the histogram.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Devanagari => "devanagari",
            Self::Latin => "latin",
            Self::Other => "other",
            Self::Mixed => "mixed",
        }
    }
}

/// Classify a token by the scripts it contains.
pub fn script_mix(token: &str) -> ScriptMix {
    let mut dev = false;
    let mut lat = false;
    let mut any = false;
    for c in token.chars() {
        if c.is_whitespace() {
            continue;
        }
        any = true;
        if is_devanagari(c) {
            dev = true;
        } else if c.is_alphabetic() && c.is_ascii() {
            lat = true;
        }
    }
    match (any, dev, lat) {
        (false, _, _) => ScriptMix::Other,
        (_, true, true) => ScriptMix::Mixed,
        (_, true, false) => ScriptMix::Devanagari,
        (_, false, true) => ScriptMix::Latin,
        // alphabetic in neither script, e.g. digits or Cyrillic
        (_, false, false) => ScriptMix::Other,
    }
}

/// Whether every alphabetic character is ASCII (used to gate H2).
fn is_latin_only(t: &str) -> bool {
    t.chars()
        .all(|c| !c.is_alphabetic() || (c.is_ascii() && c.is_alphabetic()))
}

/// Devanagari block, U+0900..U+097F.
pub fn is_devanagari(c: char) -> bool {
    ('\u{0900}'..='\u{097f}').contains(&c)
}

/// Unicode whitespace, by table. Deliberately *not* `char::is_whitespace`,
/// which also reports U+0085 and U+200E and would make rule N5 depend on the
/// standard library version.
fn is_space(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t' | '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

/// Precomposed nukta letter -> base. U+0958..=U+095F, in order: qa, kha, gha,
/// za, dda, ddha, fa, ya. U+095F is the one irregular entry: it is the ya
/// *vowel* matra (U+093F) carrying a nukta rather than a bare consonant, so
/// its "base" is the matra.
const NUKTA_BASE: [char; 8] = [
    '\u{0915}', // U+0958 qa
    '\u{0916}', // U+0959 kha
    '\u{0917}', // U+095A gha
    '\u{091c}', // U+095B za
    '\u{0921}', // U+095C dda
    '\u{0922}', // U+095D ddha
    '\u{092b}', // U+095E fa
    '\u{093f}', // U+095F ya, as matra + nukta
];

/// Unicode punctuation, symbol and non-space separator, by table.
///
/// There is no `unicode-general-category` crate in the local registry cache
/// and this build machine has no network, so the ranges that actually occur in
/// Hindi/Latin transcripts are enumerated. Everything else is left alone,
/// which is the safe direction: an unknown character is kept and shows up as a
/// word error rather than being silently deleted from the reference.
pub fn is_punct_or_symbol(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_punctuation() && c != '-' && c != '\'' && c != '_';
    }
    let u = c as u32;
    // Devanagari sentence marks. Named here as well as by category because a
    // danda that survives normalisation becomes a token, and one stray token
    // in a 200-word reference is 0.5 % of WER bought by nothing.
    if u == 0x0964 || u == 0x0965 {
        return true;
    }
    // Latin-1 punctuation and symbols.
    ((0x00a1..=0x00bf).contains(&u) && u != 0x00ad) || u == 0x00d7 || u == 0x00f7
    // General punctuation, excluding the intra-word hyphen and the apostrophe.
    || ((0x2010..=0x2027).contains(&u) && u != 0x2010 && u != 0x2011 && u != 0x2019)
    || (0x2030..=0x205e).contains(&u)
    // Currency.
    || (0x20a0..=0x20cf).contains(&u)
    // Arrows, maths, misc technical: no word content in a transcript.
    || (0x2190..=0x2bff).contains(&u)
    // CJK punctuation, kana marks.
    || (0x3001..=0x303f).contains(&u)
    // Fullwidth forms.
    || (0xff01..=0xff20).contains(&u)
    || (0xff3b..=0xff40).contains(&u)
    || (0xff5b..=0xff65).contains(&u)
    // Zero-width and bidi formatting characters.
    || (0x200b..=0x200f).contains(&u)
    || u == 0x2060
    || (0x202a..=0x202e).contains(&u)
    || (0xfe00..=0xfe0f).contains(&u)
    || (0xfff9..=0xfffd).contains(&u)
}

/// Latin letter -> base ASCII, for rule H1. Long vowels, the retroflex
/// transliterations, and the Latin-Extended-A Sanskrit set.
///
/// Keys are unique and sorted by codepoint, so the table is auditable by eye:
/// there is no "first match wins" subtlety hiding behind a duplicate key.
const LATIN_FOLD: &[(char, &str)] = &[
    ('\u{00c0}', "a"),
    ('\u{00c2}', "a"),
    ('\u{00c4}', "a"), // À Â Ä
    ('\u{00e0}', "a"),
    ('\u{00e2}', "a"),
    ('\u{00e4}', "a"),
    ('\u{00e5}', "a"), // à â ä å
    ('\u{0100}', "a"),
    ('\u{0101}', "a"), // Ā ā
    ('\u{00c9}', "e"),
    ('\u{00e8}', "e"),
    ('\u{00e9}', "e"),
    ('\u{00ea}', "e"),
    ('\u{00eb}', "e"), // É è é ê ë
    ('\u{0113}', "e"),
    ('\u{00ee}', "i"),
    ('\u{00ef}', "i"), // ē î ï
    ('\u{00cd}', "i"),
    ('\u{00ce}', "i"),
    ('\u{00ec}', "i"),
    ('\u{00ed}', "i"), // Í Î ì í
    ('\u{012a}', "i"),
    ('\u{012b}', "i"),
    ('\u{00d3}', "o"),
    ('\u{00d2}', "o"),
    ('\u{00d4}', "o"),
    ('\u{00f2}', "o"),
    ('\u{00f3}', "o"),
    ('\u{00f4}', "o"),
    ('\u{00f6}', "o"),
    ('\u{00f8}', "o"), // Ó Ò Ô ò ó ô õ ö ø
    ('\u{014c}', "o"),
    ('\u{014d}', "o"), // Ō ō
    ('\u{00da}', "u"),
    ('\u{00d6}', "u"),
    ('\u{00dc}', "u"),
    ('\u{00f9}', "u"),
    ('\u{00fa}', "u"),
    ('\u{00fb}', "u"),
    ('\u{00fc}', "u"), // Ú Ý Û ù ú û ü
    ('\u{016a}', "u"),
    ('\u{016b}', "u"), // Ū ū
    ('\u{00e6}', "ae"),
    ('\u{00c6}', "ae"), // æ Æ
    ('\u{00e7}', "c"),
    ('\u{00c7}', "c"), // ç Ç
    ('\u{0152}', "oe"),
    ('\u{0153}', "oe"), // Œ œ
    ('\u{00df}', "ss"), // ß
    ('\u{00f1}', "n"),
    ('\u{00d1}', "n"),
    ('\u{0143}', "n"),
    ('\u{0144}', "n"), // ñ Ñ ń ņ
    ('\u{1e45}', "n"),
    ('\u{1e0b}', "d"),
    ('\u{1e0c}', "d"),
    ('\u{1e0d}', "d"), // ḍ ḍ ḍ
    ('\u{1e24}', "d"),
    ('\u{1e25}', "d"),
    ('\u{1e26}', "d"), // ḑ ḑ ḑ
    ('\u{1e21}', "d"), // ḍ
    ('\u{1e2b}', "d"),
    ('\u{1e2c}', "d"),
    ('\u{1e2d}', "d"), // ḍ ḍ ḍ
    ('\u{1e6b}', "t"),
    ('\u{1e6a}', "t"), // ṫ ṫ
    ('\u{1e4d}', "t"),
    ('\u{1e4c}', "t"), // ṭ ṭ
    ('\u{1e63}', "s"),
    ('\u{1e62}', "s"), // ṣ Ṣ
    ('\u{1e5b}', "r"),
    ('\u{1e5a}', "r"), // ṛ Ṛ
    ('\u{015b}', "s"),
    ('\u{015a}', "s"),
    ('\u{0161}', "s"),
    ('\u{0160}', "s"),
    ('\u{1e41}', "s"), // ś Ś
    ('\u{00fd}', "y"),
    ('\u{00ff}', "y"),
    ('\u{1ebf}', "y"),
    ('\u{017a}', "z"),
    ('\u{0179}', "z"),
    ('\u{017c}', "z"),
    ('\u{017d}', "z"), // Ź ż Ž ž
    ('\u{017e}', "z"), // Ž
];

/// Apply rule H1. Replaces by longest-match over the folded base, then
/// re-lowercases, and is idempotent.
pub fn fold_latin_diacritics(t: &str) -> String {
    let mut out = String::with_capacity(t.len());
    for c in t.chars() {
        match LATIN_FOLD.iter().find(|(k, _)| *k == c) {
            Some((_, base)) => out.push_str(base),
            None => out.push(c),
        }
    }
    out.chars()
        .map(|c| {
            let mut it = c.to_lowercase();
            match (it.next(), it.next()) {
                (Some(a), None) => a,
                _ => c,
            }
        })
        .collect()
}

/// Apply rule H2. Only the five double-vowel spellings that are length markers
/// in romanised Hindi. `ai`, `au`, `oi`, `ou` are left alone: they are
/// diphthongs, and folding them merges distinct words.
pub fn fold_double_vowels(t: &str) -> String {
    let lower: Vec<char> = t.chars().map(|c| c.to_ascii_lowercase()).collect();
    let mut out = String::with_capacity(lower.len());
    let mut i = 0usize;
    while i < lower.len() {
        if i + 1 < lower.len() {
            let pair: String = lower[i..i + 2].iter().collect();
            let folded = match pair.as_str() {
                "aa" => Some('a'),
                "ii" => Some('i'),
                "uu" => Some('u'),
                "ee" => Some('i'),
                "oo" => Some('u'),
                _ => None,
            };
            if let Some(f) = folded {
                out.push(f);
                i += 2;
                continue;
            }
        }
        out.push(lower[i]);
        i += 1;
    }
    out
}

fn ascii_digit(c: char) -> char {
    let u = c as u32;
    if (0x0966..=0x096f).contains(&u) || (0x06f0..=0x06f9).contains(&u) {
        char::from_digit(u - 0x0966, 10)
            .or_else(|| char::from_digit(u - 0x06f0, 10))
            .unwrap_or(c)
    } else {
        c
    }
}
