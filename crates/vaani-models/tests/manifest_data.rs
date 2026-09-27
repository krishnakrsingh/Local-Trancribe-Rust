//! The manifest is data, so it is tested as data.
//!
//! Every row of the table in [`every_manifest_error_is_specific_rather_than_a_parse_failure`]
//! is a manifest a human could plausibly write by mistake, paired with the one
//! thing the resulting message must mention. The point is not that the loader
//! rejects bad input — it is that it rejects each *kind* of bad input with an
//! error that names the model and the field, because the person who has to fix
//! it is editing a text file and needs to know which line.

use vaani_models::manifest::{Manifest, SCHEMA_VERSION};

/// A manifest with one model, assembled from parts.
///
/// Built rather than patched with `String::replace`: a fixture built by
/// substitution stops testing what its name says the moment the fixture changes,
/// which is exactly how a "missing licence" test ends up testing a missing
/// comma. The two knobs a case needs are the extra file list and a set of field
/// overrides.
struct Fixture {
    schema_version: Option<&'static str>,
    licence: Option<&'static str>,
    source: &'static str,
    task: &'static str,
    access: &'static str,
    languages: &'static str,
    sample_rates: &'static str,
    size_bytes: Option<&'static str>,
    resources: Option<&'static str>,
    gpu: Option<&'static str>,
    extra_fields: &'static str,
    files: &'static [(&'static str, &'static str, u64, &'static str, bool)],
}

impl Default for Fixture {
    fn default() -> Self {
        Self {
            schema_version: Some("1"),
            licence: Some(
                r#"{ "spdx": "MIT", "name": "MIT License", "url": "https://example.org/licence" }"#,
            ),
            source: r#"{ "kind": "bundled" }"#,
            task: "asr",
            access: "open",
            languages: r#"["hi", "en"]"#,
            sample_rates: "[16000]",
            size_bytes: None,
            resources: None,
            gpu: None,
            extra_fields: "",
            files: &[(
                "weights.bin",
                "weights",
                4,
                "0000000000000000000000000000000000000000000000000000000000000000",
                true,
            )],
        }
    }
}

impl Fixture {
    fn json(&self) -> String {
        let mut files = Vec::new();
        let mut total = 0u64;
        for (path, role, bytes, sha, required) in self.files {
            files.push(format!(
                r#"{{"path": "{path}", "role": "{role}", "bytes": {bytes}, "sha256": "{sha}", "required": {required}}}"#
            ));
            if *required {
                total += bytes;
            }
        }
        let files = files.join(", ");
        let size = self.size_bytes.unwrap_or("SIZE");
        let size = size.replace("SIZE", &total.to_string());
        let opt = |v: Option<&'static str>| match v {
            Some(v) => format!(", {v}"),
            None => String::new(),
        };
        let schema = match self.schema_version {
            Some(v) => format!("\"schema_version\": {v},"),
            None => String::new(),
        };
        let licence = match self.licence {
            Some(v) => format!(r#""license": {v},"#),
            None => String::new(),
        };
        let resources = opt(self.resources);
        let gpu = opt(self.gpu);
        format!(
            r#"{{
  {schema}
  "updated": "2026-09-27",
  "models": [{{
    "id": "m1",{extra_fields}
    "engine_id": "e1",
    "version": "1.0",
    "display_name": "M1",
    "description": "a model",
    "task": "{task}",
    "architecture": "test-arch",
    "access": "{access}",
    {licence}
    "source": {source},
    "languages": {languages},
    "sample_rates": {sample_rates},
    "size_bytes": {size}{resources}{gpu},
    "files": [{files}]
  }}]
}}"#,
            task = self.task,
            access = self.access,
            source = self.source,
            languages = self.languages,
            sample_rates = self.sample_rates,
            extra_fields = self.extra_fields,
        )
    }
}

#[test]
fn the_fixture_itself_is_a_valid_manifest() {
    // Every negative case below is only meaningful if the positive one parses.
    let text = Fixture::default().json();
    let m = Manifest::parse(&text).expect("the fixture must be valid");
    assert_eq!(m.ids(), vec!["m1"]);
    assert_eq!(m.model("m1").unwrap().weights_bytes, 4);
}

#[test]
fn every_manifest_error_is_specific_rather_than_a_parse_failure() {
    // (name, fixture, substring the error must contain, why it matters)
    type Row = (&'static str, Fixture, &'static str, &'static str);
    let rows: Vec<Row> = vec![
        (
            "licence omitted entirely",
            Fixture {
                licence: None,
                ..Default::default()
            },
            "no usable licence",
            "shipping a model whose terms nobody has read is a legal problem",
        ),
        (
            "licence with no spdx",
            Fixture {
                licence: Some(r#"{ "name": "MIT", "url": "https://example.org/l" }"#),
                ..Default::default()
            },
            "spdx",
            "an identifier is what a licence badge is built from",
        ),
        (
            "licence with no url",
            Fixture {
                licence: Some(r#"{ "spdx": "MIT", "name": "MIT" }"#),
                ..Default::default()
            },
            "url",
            "there is nowhere for the user to read the terms",
        ),
        (
            "licence url is not a url",
            Fixture {
                licence: Some(r#"{ "spdx": "MIT", "name": "MIT", "url": "not a url" }"#),
                ..Default::default()
            },
            "licence URL",
            "a licence link that does not open is worse than none",
        ),
        (
            "licence url is not http",
            Fixture {
                licence: Some(r#"{ "spdx": "MIT", "name": "MIT", "url": "ftp://example.org/l" }"#),
                ..Default::default()
            },
            "licence URL",
            "the UI opens these in a browser",
        ),
        (
            "no files",
            Fixture {
                files: &[],
                ..Default::default()
            },
            "lists no files",
            "a model with no files is not a model",
        ),
        (
            "schema version 2",
            Fixture {
                schema_version: Some("2"),
                ..Default::default()
            },
            "schema version",
            "a newer manifest read by an older build would skip a checksum",
        ),
        (
            "schema version absent",
            Fixture {
                schema_version: None,
                ..Default::default()
            },
            "schema version",
            "a manifest must declare its version",
        ),
        (
            "huggingface repo is not org/name",
            Fixture {
                source: r#"{ "kind": "huggingface", "repo": "just-a-name" }"#,
                ..Default::default()
            },
            "malformed repository",
            "the URL is derived from this, so a bad repo is a bad URL",
        ),
        (
            "direct source with a malformed template",
            Fixture {
                source: r#"{ "kind": "direct", "url_template": "h t t p:/nope/{path}" }"#,
                ..Default::default()
            },
            "download URL",
            "validating the template is the only way to catch this",
        ),
        (
            "direct source with no template",
            Fixture {
                source: r#"{ "kind": "direct" }"#,
                ..Default::default()
            },
            "url_template",
            "there would be nothing to build a URL from",
        ),
        (
            "unknown source kind",
            Fixture {
                source: r#"{ "kind": "ftp" }"#,
                ..Default::default()
            },
            "unknown provider",
            "an unknown provider cannot be turned into a URL",
        ),
        (
            "unknown task",
            Fixture {
                task: "transcribing",
                ..Default::default()
            },
            "unknown task",
            "the UI groups by task",
        ),
        (
            "unknown access",
            Fixture {
                access: "free",
                ..Default::default()
            },
            "unknown access",
            "access decides whether a token is required",
        ),
        (
            "hash too short",
            Fixture {
                files: &[("weights.bin", "weights", 4, "0000", true)],
                ..Default::default()
            },
            "malformed sha256",
            "a truncated hash is the classic hand-editing mistake",
        ),
        (
            "hash is not hex",
            Fixture {
                files: &[(
                    "weights.bin",
                    "weights",
                    4,
                    "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
                    true,
                )],
                ..Default::default()
            },
            "malformed sha256",
            "base64 and hex look alike to a human",
        ),
        (
            "zero byte file",
            Fixture {
                files: &[(
                    "weights.bin",
                    "weights",
                    0,
                    "0000000000000000000000000000000000000000000000000000000000000000",
                    true,
                )],
                ..Default::default()
            },
            "impossible size",
            "no model file is zero bytes",
        ),
        (
            "file path escapes the model directory",
            Fixture {
                files: &[(
                    "../../etc/passwd",
                    "weights",
                    4,
                    "0000000000000000000000000000000000000000000000000000000000000000",
                    true,
                )],
                ..Default::default()
            },
            "'..' component",
            "a manifest is data; a path that escapes its model is an attack",
        ),
        (
            "file path uses a backslash",
            Fixture {
                files: &[(
                    "fe\\\\tables.bin",
                    "frontend",
                    4,
                    "0000000000000000000000000000000000000000000000000000000000000000",
                    true,
                )],
                ..Default::default()
            },
            "backslash",
            "manifest paths are `/`-separated so a manifest is portable",
        ),
        (
            "no languages",
            Fixture {
                languages: "[]",
                ..Default::default()
            },
            "declares no languages",
            "the language picker needs a list",
        ),
        (
            "language tag with a slash",
            Fixture {
                languages: r#"["hi/IN"]"#,
                ..Default::default()
            },
            "malformed language tag",
            "a path in a language field is a copy-paste error",
        ),
        (
            "no sample rates",
            Fixture {
                sample_rates: "[]",
                ..Default::default()
            },
            "sample rate",
            "the media stage must know what to resample to",
        ),
        (
            "zero sample rate",
            Fixture {
                sample_rates: "[0]",
                ..Default::default()
            },
            "sample rate",
            "0 Hz would divide by zero in the resampler",
        ),
        (
            "declared total disagrees with the file list",
            Fixture {
                size_bytes: Some("999"),
                ..Default::default()
            },
            "files sum to",
            "the UI shows the declared total before downloading, so it has to be right",
        ),
        (
            "recommended below minimum",
            Fixture {
                resources: Some(
                    r#""resources": { "min_vram_bytes": 2000, "rec_vram_bytes": 1000 }"#,
                ),
                ..Default::default()
            },
            "recommends",
            "a user cannot act on a recommendation below the minimum",
        ),
        (
            "bad compute capability",
            Fixture {
                gpu: Some(r#""gpu": { "compute_capability_min": "eight point six" }"#),
                ..Default::default()
            },
            "compute capability",
            "the fp16 path needs a real number here",
        ),
        (
            "directory name with a separator",
            Fixture {
                extra_fields: r#""dir": "a/b","#,
                ..Default::default()
            },
            "on-disk directory",
            "the directory is one path component, not a path",
        ),
        (
            "duplicate file path",
            Fixture {
                files: &[
                    (
                        "weights.bin",
                        "weights",
                        4,
                        "0000000000000000000000000000000000000000000000000000000000000000",
                        true,
                    ),
                    (
                        "weights.bin",
                        "config",
                        4,
                        "1111111111111111111111111111111111111111111111111111111111111111",
                        true,
                    ),
                ],
                ..Default::default()
            },
            "twice",
            "two entries for one path means one of them is never fetched",
        ),
    ];

    let mut failures = Vec::new();
    for (name, fx, needle, why) in &rows {
        match Manifest::parse(&fx.json()) {
            Ok(m) => failures.push(format!("{name}: LOADED ({:?}) -- {why}", m.ids())),
            Err(e) => {
                let msg = e.to_string();
                if !needle.is_empty() && !msg.contains(needle) {
                    failures.push(format!(
                        "{name}: {msg:?} does not mention {needle:?} -- {why}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases misbehaved:\n  {}",
        failures.len(),
        rows.len(),
        failures.join("\n  ")
    );
}

#[test]
fn a_duplicate_id_is_refused() {
    let one = Fixture::default().json();
    let m = Manifest::parse(&one).unwrap();
    let model = m.model("m1").unwrap();
    // Two entries with the same id: the primary key is not unique and
    // `model("m1")` would silently return the first.
    let two = format!(
        r#"{{"schema_version": 1, "models": [{one},{one}]}}"#,
        one = model_json_of(model)
    );
    let e = Manifest::parse(&two).unwrap_err();
    assert!(e.to_string().contains("share the id"), "{e}");
}

#[test]
fn two_models_claiming_one_directory_are_refused() {
    let text = format!(
        r#"{{"schema_version": 1, "models": [{a},{b}]}}"#,
        a = model_json("m1", "one"),
        b = model_json("m2", "one")
    );
    let e = Manifest::parse(&text).unwrap_err();
    assert!(e.to_string().contains("on-disk directory"), "{e}");
}

fn model_json(id: &str, dir: &str) -> String {
    format!(
        r#"{{"id":"{id}","dir":"{dir}","engine_id":"e1","version":"1",
        "display_name":"M","description":"d","task":"asr","architecture":"a",
        "access":"open",
        "license":{{"spdx":"MIT","name":"MIT","url":"https://example.org/l"}},
        "source":{{"kind":"bundled"}},"languages":["hi"],"sample_rates":[16000],
        "size_bytes":4,
        "files":[{{"path":"w.bin","role":"weights","bytes":4,
          "sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}]}}"#
    )
}

fn model_json_of(m: &vaani_models::manifest::Model) -> String {
    // Round-trip a validated model back to text. `Manifest` implements
    // `Serialize`, so the canonical form is available, and using it here means
    // this test does not hand-maintain a second copy of the fixture.
    let one = serde_json::json!({ "schema_version": SCHEMA_VERSION, "models": [m] });
    one["models"][0].to_string()
}

#[test]
fn a_file_with_no_required_key_is_required() {
    // The default is `true`, and it has to be: a file entry that silently
    // defaulted to optional would let a model load with a missing weights file.
    let text = r#"{"schema_version":1,"models":[{
      "id":"m1","engine_id":"e1","version":"1","display_name":"M","description":"d",
      "task":"asr","architecture":"a","access":"open",
      "license":{"spdx":"MIT","name":"MIT","url":"https://example.org/l"},
      "source":{"kind":"bundled"},"languages":["hi"],"sample_rates":[16000],
      "size_bytes":4,
      "files":[{"path":"w.bin","role":"weights","bytes":4,
        "sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]
    }]}"#;
    let m = Manifest::parse(text).unwrap();
    assert_eq!(m.model("m1").unwrap().required_paths(), vec!["w.bin"]);
}

#[test]
fn a_model_whose_files_are_all_optional_is_refused() {
    let text = r#"{"schema_version":1,"models":[{
      "id":"m1","engine_id":"e1","version":"1","display_name":"M","description":"d",
      "task":"asr","architecture":"a","access":"open",
      "license":{"spdx":"MIT","name":"MIT","url":"https://example.org/l"},
      "source":{"kind":"bundled"},"languages":["hi"],"sample_rates":[16000],
      "size_bytes":4,
      "files":[{"path":"w.bin","role":"reference","bytes":4,"required":false,
        "sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]
    }]}"#;
    let e = Manifest::parse(text).unwrap_err();
    assert!(e.to_string().contains("no required files"), "{e}");
}

#[test]
fn a_manifest_survives_a_serialise_parse_round_trip() {
    // `Manifest` implements `Serialize` so the CLI can print it and so a tool can
    // rewrite it. If a serialised manifest would not load again, both of those
    // are broken in a way nobody would notice until a user did it.
    let original = Manifest::embedded().unwrap();
    let text = serde_json::to_string_pretty(&original).expect("serialisable");
    if let Err(e) = Manifest::parse(&text) {
        let all: Vec<&str> = text.lines().collect();
        let around: Vec<String> = all
            .iter()
            .enumerate()
            .skip(60)
            .take(14)
            .map(|(i, l)| format!("{i:>4}: {l}"))
            .collect();
        panic!(
            "a serialised manifest must reload:\n{e}\n{}",
            around.join("\n")
        );
    }
    let reloaded = Manifest::parse(&text).expect("a serialised manifest must reload");
    assert_eq!(original, reloaded);
    // And the derived fields are recomputed rather than trusted from the text.
    assert_eq!(
        reloaded.model("sravaani-1.0-onnx").unwrap().weights_bytes,
        929_977_857
    );
    assert!(
        !text.contains("weights_bytes"),
        "a derived field must not be written out"
    );
    assert!(
        !text.contains("\"relative\""),
        "a derived path must not be written out"
    );
}

#[test]
fn a_misspelled_key_is_a_load_failure_not_a_silent_default() {
    // The most valuable line in the whole module. Without `deny_unknown_fields`,
    // `sze` would be ignored, `size_bytes` would be missing, and the model
    // would load with no size at all.
    let good = Fixture::default().json();
    let bad = good.replace(r#""size_bytes": 4"#, r#""sze": 4, "size_bytes": 4"#);
    let e = Manifest::parse(&bad).unwrap_err();
    assert!(e.to_string().contains("not well-formed"), "{e}");
}

#[test]
fn a_manifest_with_no_models_is_refused() {
    let e = Manifest::parse(r#"{"schema_version": 1, "models": []}"#).unwrap_err();
    assert!(e.to_string().contains("declares no models"), "{e}");
}

#[test]
fn one_bad_model_refuses_the_whole_manifest() {
    // Deliberate: a registry that loads "the models that happened to parse" is
    // how a model ships without a licence.
    let good = Fixture::default().json();
    let no_licence = Fixture {
        licence: None,
        ..Default::default()
    }
    .json();
    let text = format!(
        r#"{{"schema_version": 1, "models": [{good},{bad}]}}"#,
        good = good,
        bad = no_licence
    );
    let e = Manifest::parse(&text).unwrap_err();
    assert!(e.to_string().contains("licence"), "{e}");
}

#[test]
fn an_uppercase_hash_is_accepted_and_normalised() {
    // A human pasting from `certutil -hashfile` gets upper case. Refusing it
    // teaches people the file is fussy, which is how a checksum gets worked
    // around.
    let text = Fixture {
        files: &[(
            "w.bin",
            "weights",
            4,
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            true,
        )],
        ..Default::default()
    }
    .json();
    let m = Manifest::parse(&text).expect("upper case must load");
    assert_eq!(
        m.model("m1").unwrap().file("w.bin").unwrap().sha256,
        [0xAA; 32]
    );
}

#[test]
fn a_nested_file_path_is_allowed_and_resolved() {
    let text = Fixture {
        files: &[
            (
                "w.bin",
                "weights",
                4,
                "0000000000000000000000000000000000000000000000000000000000000000",
                true,
            ),
            (
                "fe/tables.bin",
                "frontend",
                6,
                "1111111111111111111111111111111111111111111111111111111111111111",
                true,
            ),
        ],
        ..Default::default()
    }
    .json();
    let m = Manifest::parse(&text).expect("a nested path is legitimate");
    let f = m.model("m1").unwrap().file("fe/tables.bin").unwrap();
    assert_eq!(
        f.relative,
        std::path::PathBuf::from("fe").join("tables.bin")
    );
    assert_eq!(m.model("m1").unwrap().weights_bytes, 10);
}

#[test]
fn the_same_manifest_in_toml_and_json_produces_the_same_model() {
    // A registry a human edits wants comments (TOML) and a registry a tool emits
    // wants JSON. Refusing either would be an arbitrary constraint.
    let toml = r#"
schema_version = 1
[[models]]
id = "m1"
engine_id = "e1"
version = "1.0"
display_name = "M1"
description = "a model"
task = "asr"
architecture = "test-arch"
access = "open"
languages = ["hi", "en"]
sample_rates = [16000]
size_bytes = 4
[models.license]
spdx = "MIT"
name = "MIT License"
url = "https://example.org/licence"
[models.source]
kind = "bundled"
[[models.files]]
path = "weights.bin"
role = "weights"
bytes = 4
required = true
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#;
    let from_toml = Manifest::parse(toml).unwrap();
    let from_json = Manifest::parse(&Fixture::default().json()).unwrap();
    let a = from_toml.model("m1").unwrap();
    let b = from_json.model("m1").unwrap();
    assert_eq!(a.id, b.id);
    assert_eq!(a.size_bytes, b.size_bytes);
    assert_eq!(a.files, b.files);
    assert_eq!(a.license, b.license);
    // A TOML manifest is human-editable in a way JSON is not: this one carries a
    // comment and it survives.
    let commented = format!("# a comment\n{toml}");
    assert!(Manifest::parse(&commented).is_ok());
}

#[test]
fn the_embedded_manifest_matches_the_facts_the_documentation_states() {
    // MODELS.md and ADR-004 are the authority on what SraVaani is. This test is
    // the join between that prose and the machine-readable form, so the two
    // cannot drift apart without a build failing.
    let m = Manifest::embedded().expect("the shipped manifest must be valid");
    assert_eq!(m.schema_version, SCHEMA_VERSION);
    assert_eq!(m.ids(), vec!["sravaani-1.0-onnx"]);

    let model = m.model("sravaani-1.0-onnx").unwrap();
    assert_eq!(model.engine_id, "sravaani", "the registry key in vaani-asr");
    assert_eq!(
        model.dir, "sravaani",
        "artifacts_in() looks in artifacts/sravaani"
    );
    assert_eq!(model.version, "1.0");
    assert_eq!(
        model.access.as_str(),
        "gated-token",
        "README:44,155 -- it is gated"
    );
    assert_eq!(model.license.spdx, "mit", "README:2");
    assert_eq!(model.source.provider.as_str(), "huggingface");
    assert_eq!(model.source.repo, "ARTPARK-IISc/SraVaani-1.0", "README:44");
    assert_eq!(model.source.revision, "main");
    assert_eq!(model.runtime.opset, Some(17), "ADR-004: opset 17");
    assert_eq!(model.sample_rates, vec![16000], "config.json sample_rate");
    assert_eq!(
        model.gpu.compute_capability(),
        Some((8, 6)),
        "MODELS.md: fp16 path"
    );
    assert!(model.source.needs_bearer(), "a gated repo needs the token");
    assert_eq!(
        model.source.url_for("encoder.fp16.onnx").unwrap().as_str(),
        "https://huggingface.co/ARTPARK-IISc/SraVaani-1.0/resolve/main/encoder.fp16.onnx"
    );

    // The measured facts.
    assert_eq!(
        model.size_bytes, 2_702_792_388,
        "the directory total on disk"
    );
    assert_eq!(model.weights_bytes, 929_977_857, "required files only");
    assert_eq!(model.files.len(), 10, "every file in artifacts/sravaani/");
    assert_eq!(model.file("encoder.fp16.onnx").unwrap().bytes, 886_484_931);
    assert_eq!(model.file("encoder.onnx").unwrap().bytes, 1_772_812_911);
    assert_eq!(model.file("decoder_joint.onnx").unwrap().bytes, 43_033_439);
    assert_eq!(model.file("tokenizer.model").unwrap().bytes, 325_287);
    assert_eq!(model.file("mel_window.f32").unwrap().bytes, 1_600);
    assert_eq!(model.file("mel_fb.f32").unwrap().bytes, 131_584);
    assert_eq!(model.file("preproc.json").unwrap().bytes, 426);
    assert_eq!(model.file("config.json").unwrap().bytes, 590);
    assert_eq!(model.file("model_meta.json").unwrap().bytes, 257);
    assert_eq!(model.file("artefacts.json").unwrap().bytes, 1_363);

    // The fp32 encoder is present, described, and optional: 1.65 GiB of the
    // 2.5 GiB total that no runtime path opens. It is kept so the fp16-vs-fp32
    // parity check can be re-run, so it must never become "required" and must
    // never be deleted.
    let fp32 = model.file("encoder.onnx").unwrap();
    assert!(!fp32.required);
    assert_eq!(fp32.role, vaani_models::manifest::role::REFERENCE);

    // The seven files the loader actually opens are exactly the required ones.
    let required: Vec<&str> = model.required_paths();
    assert_eq!(
        required,
        vec![
            "encoder.fp16.onnx",
            "decoder_joint.onnx",
            "tokenizer.model",
            "mel_window.f32",
            "mel_fb.f32",
            "preproc.json",
            "config.json",
        ]
    );
}
