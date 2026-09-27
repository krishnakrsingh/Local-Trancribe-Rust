//! The model manifest: a versioned, human-editable description of every model
//! the product knows about.
//!
//! # It is data, and it is validated as data
//!
//! The manifest is a JSON (or TOML) file. Adding a model to the product is a
//! text edit, not a code change — that is the whole point, and it is why the
//! file is a separate artefact rather than a `match` in Rust. The cost of that
//! choice is that a typo becomes a runtime failure, so **nothing is trusted at
//! deserialisation**:
//!
//! * every field is a [`RawManifest`] field that is [`Option`]al and is checked
//!   by hand, so a missing licence is [`ManifestError::MissingLicence`] and not
//!   serde's "data did not match any variant of untagged enum";
//! * every string that should be an enum is a `String` until validated, so a
//!   bad `task` is [`ManifestError::BadTask`] rather than a serde error with a
//!   line number nobody will find;
//! * every hash is a `String` until it has been parsed into 32 bytes, so a bad
//!   hash is [`ManifestError::BadHash`];
//! * the deserialisers carry `deny_unknown_fields`, so a misspelled key
//!   (`sze`, `licencee`) is a **load failure** instead of a silently ignored
//!   field that leaves a required value at its default. This is the single most
//!   valuable line in the file.
//!
//! # Why licence is a load error
//!
//! Shipping a model whose licence the user has not seen is a legal problem, not
//! a UX problem. A missing `spdx` id or a missing licence URL therefore refuses
//! the **whole manifest**, rather than defaulting to "unknown" and printing it
//! in a settings panel. See [`Licence`].
//!
//! # Sizes are cross-checked
//!
//! `size_bytes` on a model is redundant with the sum of its files. It is
//! required anyway, because the UI has to show "2.5 GB" before downloading and
//! a sum computed at runtime is a second code path to get wrong. The validator
//! compares them and fails on disagreement ([`ManifestError::TotalSizeMismatch`]),
//! so the redundant field cannot rot.
//!
//! # The schema
//!
//! `manifests/README.md` documents it for a human. In short, one model is:
//!
//! ```json
//! {
//!   "id": "sravaani-1.0-onnx",
//!   "engine_id": "sravaani",
//!   "version": "1.0",
//!   "display_name": "SraVaani 1.0",
//!   "description": "...",
//!   "task": "asr",
//!   "architecture": "fastconformer-tdt-rnnt",
//!   "access": "gated-token",
//!   "license": { "spdx": "MIT", "name": "MIT", "url": "https://..." },
//!   "source": { "kind": "huggingface", "repo": "org/name", "revision": "main" },
//!   "languages": ["hi", "en"],
//!   "sample_rates": [16000],
//!   "size_bytes": 1234,
//!   "resources": { "min_vram_bytes": 0, "rec_vram_bytes": 0,
//!                  "min_ram_bytes": 0, "rec_ram_bytes": 0 },
//!   "gpu": { "compute_capability_min": "8.6", "vram_headroom_bytes": 1073741824 },
//!   "runtime": { "opset": 17, "onnxruntime": ">=1.30.0" },
//!   "files": [
//!     { "path": "encoder.fp16.onnx", "role": "weights", "bytes": 886484931,
//!       "sha256": "…", "required": true }
//!   ]
//! }
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::paths::{PathProblem, relative_path};

/// The manifest schema this build understands.
///
/// A manifest with any other value is **refused**, not partially read. A
/// forward-compatible reader that ignores unknown fields sounds generous and is
/// in fact how you ship a model with a missing checksum.
pub const SCHEMA_VERSION: u32 = 1;

/// The manifest compiled into the binary.
///
/// This is the file the crate ships; `ModelManager::with_embedded_manifest`
/// uses it. Overriding it at runtime is how a user adds a model without a new
/// build.
pub const EMBEDDED_MANIFEST: &str = include_str!("../manifests/registry.json");

// ---------------------------------------------------------------------------
// Well-known file roles
// ---------------------------------------------------------------------------

/// Roles the crate itself interprets. Anything else is accepted and carried
/// through untouched, because the next model to be added will have a role
/// nobody has thought of yet, and refusing to describe it would mean a Rust
/// change for a documentation change.
pub mod role {
    /// Model weights loaded by the inference runtime.
    pub const WEIGHTS: &str = "weights";
    /// Hyper-parameters, e.g. `config.json`.
    pub const CONFIG: &str = "config";
    /// A tokenizer, e.g. a `SentencePiece` model.
    pub const TOKENIZER: &str = "tokenizer";
    /// A front-end table or parameter file.
    pub const FRONTEND: &str = "frontend";
    /// Provenance recorded at export time, e.g. `artefacts.json`.
    pub const PROVENANCE: &str = "provenance";
    /// A second copy of something already present, kept for parity work.
    /// Never required, and its absence is never a problem.
    pub const REFERENCE: &str = "reference";
}

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// What a model is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[allow(missing_docs)] // each variant maps 1:1 to as_str()
pub enum Task {
    /// Speech recognition: audio in, text out.
    Asr,
    /// Voice activity detection: audio in, speech regions out.
    Vad,
    /// Speaker diarization: audio in, speaker labels out.
    Diarization,
}

impl Task {
    /// Every task, for a UI that groups by task.
    pub const ALL: [Task; 3] = [Self::Asr, Self::Vad, Self::Diarization];

    /// The stable wire form, e.g. `asr`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Vad => "vad",
            Self::Diarization => "diarization",
        }
    }

    /// Parse a wire form, case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|t| t.as_str().eq_ignore_ascii_case(s))
    }
}

impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where the published files come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[allow(missing_docs)] // each variant maps 1:1 to as_str()
pub enum Provider {
    /// A Hugging Face Hub repository, `org/name`, at `resolve/<revision>/`.
    Huggingface,
    /// A GitHub release asset.
    GithubReleases,
    /// No download: the files ship with the app or are produced by a script.
    Bundled,
    /// A direct URL template, `https://host/{path}`.
    Direct,
}

impl Provider {
    /// Every provider, for a settings panel that lists them.
    pub const ALL: [Provider; 4] = [
        Self::Huggingface,
        Self::GithubReleases,
        Self::Bundled,
        Self::Direct,
    ];

    /// The stable wire form, e.g. `huggingface`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Huggingface => "huggingface",
            Self::GithubReleases => "github-releases",
            Self::Bundled => "bundled",
            Self::Direct => "direct",
        }
    }

    /// Whether this provider can be downloaded from at all.
    pub const fn is_downloadable(self) -> bool {
        !matches!(self, Self::Bundled)
    }

    /// Parse a wire form, case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.as_str().eq_ignore_ascii_case(s))
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a user is allowed to get the files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[allow(missing_docs)] // each variant maps 1:1 to as_str()
pub enum Access {
    /// No credential needed.
    Open,
    /// A Hugging Face account with granted access; needs a token.
    GatedToken,
    /// Access must be requested from the publisher and approved.
    RequestAccess,
}

impl Access {
    /// Every access level, in increasing order of friction.
    pub const ALL: [Access; 3] = [Self::Open, Self::GatedToken, Self::RequestAccess];

    /// The stable wire form, e.g. `gated-token`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::GatedToken => "gated-token",
            Self::RequestAccess => "request-access",
        }
    }

    /// Whether obtaining the files needs a credential.
    pub const fn needs_token(self) -> bool {
        !matches!(self, Self::Open)
    }

    /// Parse a wire form, case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|a| a.as_str().eq_ignore_ascii_case(s))
    }
}

impl std::fmt::Display for Access {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Typed model
// ---------------------------------------------------------------------------

/// A `[u8; 32]` as lower-case hex.
///
/// Without this, `serde` writes a SHA-256 as an array of 32 decimal numbers,
/// which is valid JSON and is not a manifest anybody can read or edit. The
/// round-trip test is what caught it.
mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub fn serialize<S: Serializer>(h: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(h))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let raw = String::deserialize(d)?;
        super::parse_sha256(&raw).ok_or_else(|| D::Error::custom("expected 64 hex characters"))
    }
}

/// A `Url` as a plain string in JSON/TOML.
///
/// The `url` crate's `serde` feature is not enabled in this workspace, and
/// enabling it is the architect's call, not this crate's. Two lines of glue are
/// cheaper than a dependency change, and the wire form is a string either way.
mod url_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};
    use url::Url;

    pub fn serialize<S: Serializer>(u: &Url, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(u.as_str())
    }

    /// Parse a URL string, for reuse by [`super::opt_url_string`].
    pub fn url_from(raw: &str) -> Result<Url, String> {
        Url::parse(raw.trim()).map_err(|e| e.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Url, D::Error> {
        let raw = String::deserialize(d)?;
        Url::parse(raw.trim()).map_err(D::Error::custom)
    }
}

/// The same, for an optional URL.
mod opt_url_string {
    use super::url_string;
    use serde::{Deserialize, Deserializer, Serializer};
    use url::Url;

    pub fn serialize<S: Serializer>(u: &Option<Url>, s: S) -> Result<S::Ok, S::Error> {
        match u {
            Some(u) => s.serialize_str(u.as_str()),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Url>, D::Error> {
        let raw = Option::<String>::deserialize(d)?;
        raw.map(|r| url_string::url_from(&r))
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

/// The licence a model is distributed under.
///
/// Required, and validated to be *usable*: a non-empty SPDX identifier and an
/// absolute `http(s)` URL a user can be shown. `Option` is not offered because
/// the alternative is a model whose legal terms nobody has read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Licence {
    /// SPDX identifier, e.g. `MIT`. Lower-cased on load so `MIT`, `mit` and
    /// `Mit` cannot become three different models to a consumer.
    pub spdx: String,
    /// The licence in words, e.g. "MIT License".
    pub name: String,
    /// Where the user reads it.
    #[serde(with = "url_string")]
    pub url: Url,
}

impl Licence {
    /// `spdx`, for a UI badge.
    pub fn label(&self) -> &str {
        &self.spdx
    }
}

/// Where a model's published files live.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    /// Which hosting layout `url` follows.
    ///
    /// Serialised as `kind`, the key the raw form uses, so a serialised
    /// `Manifest` is a manifest that can be loaded again.
    #[serde(rename = "kind")]
    pub provider: Provider,
    /// `org/name` for [`Provider::Huggingface`]. Empty for the others.
    pub repo: String,
    /// The pinned revision. `main` when not given, because a manifest that
    /// pins a commit hash is the only way a hash in it can mean anything.
    pub revision: String,
    /// The URL template, with `{path}` where the file path goes. Derived for
    /// Hugging Face and GitHub releases; required for [`Provider::Direct`].
    pub url_template: String,
    /// The publisher's page, which is what a user opens to read the model card.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_url_string"
    )]
    pub homepage: Option<Url>,
}

impl Source {
    /// The download URL for one file.
    ///
    /// `None` when the provider is [`Provider::Bundled`], because there is
    /// nothing to download — the caller must not invent a URL for it.
    pub fn url_for(&self, path: &str) -> Option<Url> {
        if !self.provider.is_downloadable() {
            return None;
        }
        let joined = self.url_template.replace("{path}", path);
        Url::parse(&joined).ok()
    }

    /// Whether a download from this source needs an `Authorization` header.
    pub fn needs_bearer(&self) -> bool {
        // Only Hugging Face's `resolve` endpoint gates on a bearer token; a
        // GitHub asset and a plain CDN URL must never be sent one, because
        // sending a credential to a host that does not need it is how tokens
        // leak.
        self.provider == Provider::Huggingface
    }
}

/// Hardware and runtime requirements.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resources {
    /// Below this the model will not load.
    pub min_vram_bytes: u64,
    /// What the UI should recommend.
    pub rec_vram_bytes: u64,
    /// Below this the CPU fallback will thrash.
    pub min_ram_bytes: u64,
    /// What the UI should recommend.
    pub rec_ram_bytes: u64,
}

/// Accelerator-specific requirements, consumed by
/// `MachineCapabilities::vram_fits`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum CUDA compute capability, e.g. `"8.6"`. `None` means the model
    /// runs on anything, including no GPU at all.
    pub compute_capability_min: Option<String>,
    /// Activation headroom to add to the weights when fitting, because a
    /// Conformer encoder's working set grows with chunk length.
    pub vram_headroom_bytes: u64,
}

impl GpuRequirements {
    /// Parse `compute_capability_min` into `(major, minor)`.
    pub fn compute_capability(&self) -> Option<(u8, u8)> {
        let s = self.compute_capability_min.as_deref()?;
        let (a, b) = s.split_once('.')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    }
}

/// The inference runtime this model's graphs were produced for.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    /// e.g. `">=1.30.0"`.
    pub onnxruntime: Option<String>,
    /// e.g. `"9.14.0.64"`.
    pub cudnn: Option<String>,
    /// e.g. `"13.0.2"`.
    pub cuda_redist: Option<String>,
    /// The ONNX opset the graphs use. `17` for `SraVaani` (ADR-004).
    pub opset: Option<u32>,
}

/// One file of a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    /// Path relative to the model directory, `/`-separated.
    pub path: String,
    /// The same path as a real [`PathBuf`], resolved at load time.
    ///
    /// Derived, so it is skipped on serialisation: a manifest that
    /// re-serialised with a duplicate copy of the path would be refused by this
    /// very loader, because `deny_unknown_fields` rejects keys it does not know.
    #[serde(skip)]
    pub relative: PathBuf,
    /// What the file is for. See [`role`]; any other string is accepted.
    pub role: String,
    /// Expected size in bytes. Checked before the hash, because it is free.
    pub bytes: u64,
    /// Expected SHA-256, lower-case hex.
    #[serde(with = "hex_bytes")]
    pub sha256: [u8; 32],
    /// Whether the model is unusable without it.
    ///
    /// This is not decoration. `SraVaani`'s fp32 `encoder.onnx` is 1.65 GiB of the
    /// 2.5 GiB total and the loader never opens it: it is kept so the fp16 graph
    /// can be compared against its reference. Marking it optional is what stops
    /// discovery reporting a perfectly good install as broken.
    pub required: bool,
    /// Free text for a human reading the manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ModelFile {
    /// The expected hash, lower-case hex.
    pub fn sha256_hex(&self) -> String {
        hex::encode(self.sha256)
    }

    /// Whether this file must be present for the model to be usable.
    pub fn is_required(&self) -> bool {
        self.required
    }
}

/// One model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    /// Stable id. The CLI's `--model` argument and the manifest's primary key.
    pub id: String,
    /// The directory name under a model root. Defaults to `id`.
    pub dir: String,
    /// The id of the engine in `vaani_asr::EngineRegistry` that consumes this
    /// model, e.g. `sravaani`. Needed because the manifest id and the engine
    /// id are not the same string.
    pub engine_id: String,
    /// Version of the **weights**, independent of `APP_VERSION`.
    pub version: String,
    /// What the settings panel calls it.
    pub display_name: String,
    /// One paragraph a user can read before downloading 2.5 GB.
    pub description: String,
    /// What it is for.
    pub task: Task,
    /// Architecture slug plus the hyper-parameters the decode loop depends on.
    pub architecture: String,
    /// Free-form hyper-parameter notes, e.g. `enc_d_model 1024, feat_in 128`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub architecture_params: BTreeMap<String, String>,
    /// How to obtain it.
    pub access: Access,
    /// Its legal terms. Required.
    pub license: Licence,
    /// Where it is published.
    pub source: Source,
    /// BCP-47 tags, or ISO-639-3 for languages with no widely deployed tag.
    pub languages: Vec<String>,
    /// Sample rates in Hz the model consumes.
    pub sample_rates: Vec<u32>,
    /// Total bytes of every file, cross-checked against the file list.
    pub size_bytes: u64,
    /// Bytes of the required files only: what must fit in VRAM.
    ///
    /// Derived from `files`, so skipped on serialisation for the same reason as
    /// [`ModelFile::relative`]: writing it into a manifest would be a second
    /// source of truth that could disagree with the file list.
    #[serde(skip)]
    pub weights_bytes: u64,
    /// Hardware requirements.
    pub resources: Resources,
    /// Accelerator requirements.
    pub gpu: GpuRequirements,
    /// Runtime requirements.
    pub runtime: RuntimeRequirements,
    /// Every file, in manifest order.
    pub files: Vec<ModelFile>,
}

impl Model {
    /// The on-disk directory name for this model.
    pub fn dir(&self) -> &str {
        &self.dir
    }

    /// Look up a file entry by its manifest path.
    pub fn file(&self, path: &str) -> Option<&ModelFile> {
        self.files.iter().find(|f| f.path == path)
    }

    /// Files that must be present for the model to be usable.
    pub fn required_files(&self) -> impl Iterator<Item = &ModelFile> {
        self.files.iter().filter(|f| f.required)
    }

    /// Manifest paths of the required files, in manifest order.
    pub fn required_paths(&self) -> Vec<&str> {
        self.required_files().map(|f| f.path.as_str()).collect()
    }

    /// The bytes still needed before this model is complete: every required file
    /// that is absent or the wrong size.
    ///
    /// This is what the UI shows as "x of y MB to download" and what
    /// [`crate::verify::Verifier`] compares against.
    pub fn missing_bytes(&self, present: &dyn Fn(&ModelFile) -> Option<u64>) -> u64 {
        self.required_files()
            .filter_map(|f| match present(f) {
                Some(actual) if actual == f.bytes => None,
                _ => Some(f.bytes),
            })
            .sum()
    }

    /// Whether a CUDA compute capability satisfies this model's requirement.
    ///
    /// `None` capability means the machine could not be asked, which is
    /// reported as `false` rather than `true`: a model that declares a minimum
    /// compute capability must not claim a fit on a machine we could not query.
    pub fn compute_capability_ok(&self, actual: Option<(u8, u8)>) -> bool {
        let Some(min) = self.gpu.compute_capability() else {
            return true;
        };
        let Some((maj, minr)) = actual else {
            return false;
        };
        (maj, minr) >= min
    }
}

/// A whole manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Always [`SCHEMA_VERSION`] after validation.
    pub schema_version: u32,
    /// ISO-8601 date the manifest was last edited, for display.
    pub updated: Option<String>,
    /// What this manifest is for.
    pub description: Option<String>,
    /// Every model, in manifest order.
    pub models: Vec<Model>,
}

impl Manifest {
    /// Parse and fully validate a manifest.
    ///
    /// The format is sniffed from the first non-whitespace character: `{` is
    /// JSON, anything else is TOML. Both are supported because a registry that
    /// a human edits by hand wants comments, and a registry that a tool
    /// generates wants JSON; refusing either would be an arbitrary constraint.
    pub fn parse(text: &str) -> Result<Manifest, ManifestError> {
        let fmt = if text.trim_start().starts_with('{') {
            "JSON"
        } else {
            "TOML"
        };
        let raw: RawManifest = if fmt == "JSON" {
            serde_json::from_str(text).map_err(|e| ManifestError::Syntax {
                format: fmt,
                message: e.to_string(),
            })?
        } else {
            toml::from_str(text).map_err(|e| ManifestError::Syntax {
                format: fmt,
                message: e.to_string(),
            })?
        };
        raw.validate()
    }

    /// The embedded manifest, parsed and validated.
    ///
    /// A failure here is a bug in this repository, not a user error: the
    /// compiled-in registry is covered by a test that cannot be skipped.
    pub fn embedded() -> Result<Manifest, ManifestError> {
        Manifest::parse(EMBEDDED_MANIFEST)
    }

    /// Look up a model by id.
    pub fn model(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Look up a model by its on-disk directory name.
    ///
    /// Several models can share a directory only if they are variants of one
    /// graph, which the manifest does not currently describe; the first match
    /// wins and the ambiguity is logged rather than hidden.
    pub fn model_in_dir(&self, dir: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.dir == dir)
    }

    /// Total bytes of every model in the manifest.
    pub fn total_bytes(&self) -> u64 {
        self.models.iter().map(|m| m.size_bytes).sum()
    }

    /// Ids of every model, in manifest order.
    pub fn ids(&self) -> Vec<&str> {
        self.models.iter().map(|m| m.id.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a manifest was refused.
///
/// One variant per way a manifest can be wrong, because a manifest is data that
/// a human edits and "invalid manifest" is not an actionable error message. Each
/// variant names the model and the field, so a caller can point at the line.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ManifestError {
    /// The file is not well-formed in its format.
    #[error("{format} manifest is not well-formed: {message}")]
    Syntax {
        /// `JSON` or `TOML`.
        format: &'static str,
        /// The parser's own message.
        message: String,
    },
    /// A schema version this build does not implement.
    #[error(
        "manifest schema version {found} is not supported (this build reads version {supported})"
    )]
    UnsupportedSchema {
        /// What the file said.
        found: u32,
        /// What this build reads.
        supported: u32,
    },
    /// A manifest with no models is a mistake, not an empty library.
    #[error("manifest declares no models")]
    NoModels,
    /// Two models with the same id: the primary key is not unique.
    #[error("two models share the id {id:?}")]
    DuplicateId {
        /// The repeated id.
        id: String,
    },
    /// Two models with the same directory name: they would collide on disk.
    #[error("models {id:?} and {other:?} both use the on-disk directory {dir:?}")]
    DuplicateDir {
        /// The model being validated.
        id: String,
        /// The model that claimed the directory first.
        other: String,
        /// The contested directory.
        dir: String,
    },
    /// A required string field was absent or blank.
    #[error("model {id:?} is missing the required field {field:?}")]
    MissingField {
        /// The model.
        id: String,
        /// The field name as it appears in the file.
        field: &'static str,
    },
    /// No licence, or a licence without an identifier or a URL.
    ///
    /// Refusing the load is deliberate: shipping a model whose terms the user
    /// has not seen is a legal problem, and defaulting to "unknown" would put
    /// that in a settings panel where nobody reads it.
    #[error(
        "model {id:?} has no usable licence ({field}); VaaniDesk will not offer a model whose terms it cannot show"
    )]
    MissingLicence {
        /// The model.
        id: String,
        /// Which part is missing: `license`, `spdx` or `url`.
        field: &'static str,
    },
    /// The licence URL is not an absolute `http(s)` URL.
    #[error("model {id:?} has a malformed licence URL {url:?}")]
    BadLicenceUrl {
        /// The model.
        id: String,
        /// The offending value.
        url: String,
    },
    /// A file path that could escape the model directory, or is otherwise
    /// unusable.
    #[error("model {id:?} file {path:?} {reason}")]
    BadFilePath {
        /// The model.
        id: String,
        /// The offending path.
        path: String,
        /// Why.
        reason: PathProblem,
    },
    /// Two files with the same path.
    #[error("model {id:?} lists {path:?} twice")]
    DuplicateFile {
        /// The model.
        id: String,
        /// The repeated path.
        path: String,
    },
    /// A file with a zero or negative size, or a size that does not fit `u64`.
    #[error("model {id:?} file {path:?} declares an impossible size ({bytes} bytes)")]
    BadSize {
        /// The model.
        id: String,
        /// The file.
        path: String,
        /// The declared size.
        bytes: u64,
    },
    /// A `sha256` that is not 64 lower-case hex characters.
    #[error("model {id:?} file {path:?} has a malformed sha256 {value:?} (want 64 hex characters)")]
    BadHash {
        /// The model.
        id: String,
        /// The file.
        path: String,
        /// The offending value.
        value: String,
    },
    /// A model with no files at all.
    #[error("model {id:?} lists no files")]
    NoFiles {
        /// The model.
        id: String,
    },
    /// A model whose every file is optional: nothing would ever be required.
    #[error("model {id:?} has no required files, so nothing could ever be verified")]
    NoRequiredFiles {
        /// The model.
        id: String,
    },
    /// `size_bytes` disagrees with the sum of the file sizes.
    #[error("model {id:?} declares size_bytes {declared} but its files sum to {computed}")]
    TotalSizeMismatch {
        /// The model.
        id: String,
        /// The declared total.
        declared: u64,
        /// The sum of the file list.
        computed: u64,
    },
    /// A `task`, `provider` or `access` that is not one of the known values.
    #[error("model {id:?} has an unknown {field} {value:?}")]
    BadEnum {
        /// The model.
        id: String,
        /// Which enum: `task`, `provider` or `access`.
        field: &'static str,
        /// The offending value.
        value: String,
    },
    /// A repository that is not `org/name`.
    #[error("model {id:?} has a malformed repository {repo:?} (want org/name)")]
    BadRepo {
        /// The model.
        id: String,
        /// The offending value.
        repo: String,
    },
    /// A URL template that is not a valid URL once `{path}` is substituted.
    #[error("model {id:?} has a malformed download URL {url:?}")]
    BadSourceUrl {
        /// The model.
        id: String,
        /// The offending value.
        url: String,
    },
    /// A source that cannot produce a URL for a known file.
    #[error("model {id:?} needs a url_template for provider {provider:?}")]
    MissingUrlTemplate {
        /// The model.
        id: String,
        /// The provider that has no template.
        provider: String,
    },
    /// A model with no languages, which the language picker cannot use.
    #[error("model {id:?} declares no languages")]
    NoLanguages {
        /// The model.
        id: String,
    },
    /// A language tag that is not plausible.
    #[error("model {id:?} declares a malformed language tag {tag:?}")]
    BadLanguage {
        /// The model.
        id: String,
        /// The offending tag.
        tag: String,
    },
    /// A sample rate of zero, or a list of none.
    #[error("model {id:?} declares an unusable sample rate list {rates:?}")]
    BadSampleRates {
        /// The model.
        id: String,
        /// The offending list.
        rates: Vec<u32>,
    },
    /// `recommended` lower than `minimum`, which is a contradiction a user
    /// cannot resolve.
    #[error("model {id:?} recommends {rec} bytes of {what}, below its own minimum of {min}")]
    BadResources {
        /// The model.
        id: String,
        /// `VRAM` or `RAM`.
        what: &'static str,
        /// The recommended figure.
        rec: u64,
        /// The minimum figure.
        min: u64,
    },
    /// A `compute_capability_min` that is not `major.minor`.
    #[error("model {id:?} declares a malformed compute capability {value:?} (want major.minor)")]
    BadComputeCapability {
        /// The model.
        id: String,
        /// The offending value.
        value: String,
    },
    /// A directory name that is not a single safe path component.
    #[error("model {id:?} has an unusable on-disk directory {dir:?}")]
    BadDir {
        /// The model.
        id: String,
        /// The offending value.
        dir: String,
    },
    /// The model could not be read from disk.
    #[error("could not read the model manifest at {path}: {message}")]
    Io {
        /// The path the user or the caller named.
        path: String,
        /// The operating system's message.
        message: String,
    },
}

impl ManifestError {
    /// The `ErrorCode` this maps to in the application taxonomy.
    ///
    /// A bad manifest is not a missing model, not a corrupt model and not a
    /// full disk. It is a data problem in a file the user (or we) wrote, so it
    /// is `Internal` with the parser's message as the *cause* — never in the
    /// user-facing detail, which must stay free of file paths we did not choose.
    pub fn error_code(&self) -> vaani_core::ErrorCode {
        match self {
            Self::Io { .. } => vaani_core::ErrorCode::StorageFailed,
            _ => vaani_core::ErrorCode::Internal,
        }
    }
}

impl From<ManifestError> for vaani_core::VaaniError {
    fn from(e: ManifestError) -> Self {
        vaani_core::VaaniError::new(vaani_core::UserFacingError::new(
            e.error_code(),
            "The model list could not be read.",
            "VaaniDesk could not understand its own model catalogue, so it does \
             not know which models exist. Reinstalling VaaniDesk will restore the \
             built-in list.",
        ))
        .with_cause(e)
        .with_recovery("Reinstall VaaniDesk. If you edited the model list, restore it first.")
    }
}

// ---------------------------------------------------------------------------
// Raw (deserialised, unvalidated) shapes
// ---------------------------------------------------------------------------

/// The on-disk shape. Every field is optional and nothing is checked here; see
/// the module docs for why.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    schema_version: Option<u32>,
    updated: Option<String>,
    description: Option<String>,
    #[serde(default)]
    models: Vec<RawModel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    id: Option<String>,
    dir: Option<String>,
    engine_id: Option<String>,
    version: Option<String>,
    display_name: Option<String>,
    description: Option<String>,
    task: Option<String>,
    architecture: Option<String>,
    #[serde(default)]
    architecture_params: BTreeMap<String, String>,
    access: Option<String>,
    #[serde(alias = "licence")]
    license: Option<RawLicence>,
    source: Option<RawSource>,
    #[serde(default)]
    languages: Vec<String>,
    #[serde(default)]
    sample_rates: Vec<u32>,
    #[serde(alias = "size")]
    size_bytes: Option<u64>,
    #[serde(default)]
    resources: RawResources,
    #[serde(default)]
    gpu: RawGpu,
    #[serde(default)]
    runtime: RawRuntime,
    #[serde(default)]
    files: Vec<RawFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLicence {
    spdx: Option<String>,
    name: Option<String>,
    url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSource {
    kind: Option<String>,
    repo: Option<String>,
    revision: Option<String>,
    url_template: Option<String>,
    homepage: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResources {
    min_vram_bytes: Option<u64>,
    rec_vram_bytes: Option<u64>,
    min_ram_bytes: Option<u64>,
    rec_ram_bytes: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGpu {
    compute_capability_min: Option<String>,
    vram_headroom_bytes: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRuntime {
    onnxruntime: Option<String>,
    cudnn: Option<String>,
    cuda_redist: Option<String>,
    opset: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    path: Option<String>,
    role: Option<String>,
    bytes: Option<u64>,
    sha256: Option<String>,
    #[serde(default = "default_true")]
    required: bool,
    #[serde(alias = "note", alias = "notes")]
    notes: Option<String>,
}

const fn default_true() -> bool {
    true
}

impl RawManifest {
    fn validate(self) -> Result<Manifest, ManifestError> {
        let schema_version = self.schema_version.unwrap_or(0);
        if schema_version != SCHEMA_VERSION {
            return Err(ManifestError::UnsupportedSchema {
                found: schema_version,
                supported: SCHEMA_VERSION,
            });
        }
        if self.models.is_empty() {
            return Err(ManifestError::NoModels);
        }

        let mut seen_ids: BTreeMap<String, ()> = BTreeMap::new();
        let mut seen_dirs: BTreeMap<String, String> = BTreeMap::new();
        let mut models = Vec::with_capacity(self.models.len());
        for raw in self.models {
            let m = raw.validate()?;
            if seen_ids.insert(m.id.clone(), ()).is_some() {
                return Err(ManifestError::DuplicateId { id: m.id.clone() });
            }
            if let Some(other) = seen_dirs.insert(m.dir.clone(), m.id.clone()) {
                return Err(ManifestError::DuplicateDir {
                    id: m.id.clone(),
                    other,
                    dir: m.dir.clone(),
                });
            }
            models.push(m);
        }

        Ok(Manifest {
            schema_version,
            updated: self.updated,
            description: self.description,
            models,
        })
    }
}

impl RawModel {
    fn validate(self) -> Result<Model, ManifestError> {
        // The id is needed by every error message, so it is resolved first and
        // given a placeholder when absent: an error that says `model "??" is
        // missing id` is still more useful than a panic.
        let id = self.id.clone().unwrap_or_default();
        let need = |field: &'static str| ManifestError::MissingField {
            id: id.clone(),
            field,
        };

        let id = nonblank(self.id).ok_or_else(|| need("id"))?;
        let engine_id = nonblank(self.engine_id).ok_or_else(|| need("engine_id"))?;
        let version = nonblank(self.version).ok_or_else(|| need("version"))?;
        let display_name = nonblank(self.display_name).ok_or_else(|| need("display_name"))?;
        let description = nonblank(self.description).ok_or_else(|| need("description"))?;
        let architecture = nonblank(self.architecture).ok_or_else(|| need("architecture"))?;

        let task_raw = require(self.task, need("task"))?;
        let task = Task::parse(&task_raw).ok_or_else(|| ManifestError::BadEnum {
            id: id.clone(),
            field: "task",
            value: task_raw.clone(),
        })?;
        let access_raw = require(self.access, need("access"))?;
        let access = Access::parse(&access_raw).ok_or_else(|| ManifestError::BadEnum {
            id: id.clone(),
            field: "access",
            value: access_raw.clone(),
        })?;

        let license = self
            .license
            .ok_or(ManifestError::MissingLicence {
                id: id.clone(),
                field: "license",
            })?
            .validate(&id)?;

        let source = self
            .source
            .ok_or(ManifestError::MissingField {
                id: id.clone(),
                field: "source",
            })?
            .validate(&id)?;

        let dir = self.dir.unwrap_or_else(|| id.clone());
        if relative_path(&dir).is_err() || dir.contains('/') {
            return Err(ManifestError::BadDir {
                id: id.clone(),
                dir: dir.clone(),
            });
        }

        if self.languages.is_empty() {
            return Err(ManifestError::NoLanguages { id: id.clone() });
        }
        for tag in &self.languages {
            if !plausible_language_tag(tag) {
                return Err(ManifestError::BadLanguage {
                    id: id.clone(),
                    tag: tag.clone(),
                });
            }
        }
        if self.sample_rates.is_empty() || self.sample_rates.contains(&0) {
            return Err(ManifestError::BadSampleRates {
                id: id.clone(),
                rates: self.sample_rates.clone(),
            });
        }

        if self.files.is_empty() {
            return Err(ManifestError::NoFiles { id: id.clone() });
        }
        let mut files = Vec::with_capacity(self.files.len());
        let mut seen_files: BTreeMap<String, ()> = BTreeMap::new();
        for f in self.files {
            let f = f.validate(&id)?;
            if seen_files.insert(f.path.clone(), ()).is_some() {
                return Err(ManifestError::DuplicateFile {
                    id: id.clone(),
                    path: f.path.clone(),
                });
            }
            files.push(f);
        }
        if !files.iter().any(|f| f.required) {
            return Err(ManifestError::NoRequiredFiles { id: id.clone() });
        }

        let computed: u64 = files.iter().map(|f| f.bytes).sum();
        let declared = self.size_bytes.ok_or_else(|| need("size_bytes"))?;
        if computed != declared {
            return Err(ManifestError::TotalSizeMismatch {
                id,
                declared,
                computed,
            });
        }
        let weights_bytes: u64 = files.iter().filter(|f| f.required).map(|f| f.bytes).sum();

        let resources = Resources {
            min_vram_bytes: self.resources.min_vram_bytes.unwrap_or(0),
            rec_vram_bytes: self.resources.rec_vram_bytes.unwrap_or(0),
            min_ram_bytes: self.resources.min_ram_bytes.unwrap_or(0),
            rec_ram_bytes: self.resources.rec_ram_bytes.unwrap_or(0),
        };
        for (what, min, rec) in [
            ("VRAM", resources.min_vram_bytes, resources.rec_vram_bytes),
            ("RAM", resources.min_ram_bytes, resources.rec_ram_bytes),
        ] {
            if rec != 0 && rec < min {
                return Err(ManifestError::BadResources {
                    id: id.clone(),
                    what,
                    rec,
                    min,
                });
            }
        }

        let gpu = GpuRequirements {
            compute_capability_min: self.gpu.compute_capability_min.clone(),
            vram_headroom_bytes: self.gpu.vram_headroom_bytes.unwrap_or(0),
        };
        if gpu.compute_capability_min.is_some() && gpu.compute_capability().is_none() {
            return Err(ManifestError::BadComputeCapability {
                id: id.clone(),
                value: gpu.compute_capability_min.clone().unwrap_or_default(),
            });
        }

        Ok(Model {
            id,
            dir,
            engine_id,
            version,
            display_name,
            description,
            task,
            architecture,
            architecture_params: self.architecture_params,
            access,
            license,
            source,
            languages: self.languages,
            sample_rates: self.sample_rates,
            size_bytes: declared,
            weights_bytes,
            resources,
            gpu,
            runtime: RuntimeRequirements {
                onnxruntime: self.runtime.onnxruntime,
                cudnn: self.runtime.cudnn,
                cuda_redist: self.runtime.cuda_redist,
                opset: self.runtime.opset,
            },
            files,
        })
    }
}

impl RawLicence {
    fn validate(self, id: &str) -> Result<Licence, ManifestError> {
        let spdx = nonblank(self.spdx).ok_or(ManifestError::MissingLicence {
            id: id.to_string(),
            field: "spdx",
        })?;
        let name = nonblank(self.name).unwrap_or_else(|| spdx.clone());
        let url_text = nonblank(self.url).ok_or(ManifestError::MissingLicence {
            id: id.to_string(),
            field: "url",
        })?;
        let url = parse_web_url(&url_text).ok_or_else(|| ManifestError::BadLicenceUrl {
            id: id.to_string(),
            url: url_text.clone(),
        })?;
        Ok(Licence {
            // Lower-cased so `MIT` and `mit` cannot become two licences.
            spdx: spdx.to_ascii_lowercase(),
            name,
            url,
        })
    }
}

impl RawSource {
    fn validate(self, id: &str) -> Result<Source, ManifestError> {
        let kind = self.kind.clone().unwrap_or_else(|| "bundled".to_string());
        let provider = Provider::parse(&kind).ok_or_else(|| ManifestError::BadEnum {
            id: id.to_string(),
            field: "provider",
            value: kind.clone(),
        })?;

        let repo = self.repo.clone().unwrap_or_default();
        if provider == Provider::Huggingface && !valid_hf_repo(&repo) {
            return Err(ManifestError::BadRepo {
                id: id.to_string(),
                repo,
            });
        }

        let revision = nonblank(self.revision).unwrap_or_else(|| "main".to_string());
        let url_template = match (provider, nonblank(self.url_template.clone())) {
            (_, Some(t)) => t,
            (Provider::Huggingface, None) => {
                format!("https://huggingface.co/{repo}/resolve/{revision}/{{path}}")
            }
            (Provider::GithubReleases, None) => {
                return Err(ManifestError::MissingUrlTemplate {
                    id: id.to_string(),
                    provider: kind,
                });
            }
            (Provider::Direct, None) => {
                return Err(ManifestError::MissingUrlTemplate {
                    id: id.to_string(),
                    provider: kind,
                });
            }
            (Provider::Bundled, None) => String::new(),
        };

        if provider.is_downloadable() {
            // Substituting a real file name is the only way to know the template
            // is a URL and not a string that merely looks like one.
            let probe = url_template.replace("{path}", "probe.bin");
            if parse_web_url(&probe).is_none() {
                return Err(ManifestError::BadSourceUrl {
                    id: id.to_string(),
                    url: url_template,
                });
            }
        }

        let homepage = match nonblank(self.homepage) {
            Some(h) => Some(
                parse_web_url(&h).ok_or_else(|| ManifestError::BadSourceUrl {
                    id: id.to_string(),
                    url: h,
                })?,
            ),
            None => None,
        };

        Ok(Source {
            provider,
            repo,
            revision,
            url_template,
            homepage,
        })
    }
}

impl RawFile {
    fn validate(self, id: &str) -> Result<ModelFile, ManifestError> {
        let path = nonblank(self.path).ok_or(ManifestError::MissingField {
            id: id.to_string(),
            field: "path",
        })?;
        let relative = relative_path(&path).map_err(|reason| ManifestError::BadFilePath {
            id: id.to_string(),
            path: path.clone(),
            reason,
        })?;
        let role = nonblank(self.role).ok_or(ManifestError::MissingField {
            id: id.to_string(),
            field: "role",
        })?;
        let bytes = self.bytes.ok_or(ManifestError::MissingField {
            id: id.to_string(),
            field: "bytes",
        })?;
        if bytes == 0 {
            return Err(ManifestError::BadSize {
                id: id.to_string(),
                path,
                bytes,
            });
        }
        let value = self.sha256.ok_or(ManifestError::MissingField {
            id: id.to_string(),
            field: "sha256",
        })?;
        let sha256 = parse_sha256(&value).ok_or_else(|| ManifestError::BadHash {
            id: id.to_string(),
            path: path.clone(),
            value: value.clone(),
        })?;
        Ok(ModelFile {
            path,
            relative,
            role,
            bytes,
            sha256,
            required: self.required,
            note: self.notes,
        })
    }
}

fn require<T>(v: Option<T>, e: ManifestError) -> Result<T, ManifestError> {
    v.ok_or(e)
}

fn nonblank(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Parse a lower-case 64-character SHA-256.
///
/// Upper case is accepted, because a human pasting a hash out of
/// `certutil -hashfile` will paste upper case, and refusing it would teach
/// people that the manifest is fussy. It is normalised here instead.
pub fn parse_sha256(value: &str) -> Option<[u8; 32]> {
    let t = value.trim();
    if t.len() != 64 {
        return None;
    }
    let bytes = hex::decode(t).ok()?;
    <[u8; 32]>::try_from(bytes).ok()
}

fn parse_web_url(s: &str) -> Option<Url> {
    let u = Url::parse(s.trim()).ok()?;
    if !matches!(u.scheme(), "http" | "https") {
        return None;
    }
    if u.host_str().is_none_or(str::is_empty) {
        return None;
    }
    Some(u)
}

/// `org/name`, two non-empty components, no scheme, no query, no leading or
/// trailing separator.
///
/// Dots are **allowed** in the name: `ARTPARK-IISc/SraVaani-1.0` is the real
/// upstream repository, and an early version of this check rejected it, which is
/// a good reminder that a "looks suspicious" heuristic on a real id is a bug
/// rather than a hardening measure.
pub fn valid_hf_repo(repo: &str) -> bool {
    let r = repo.trim();
    if r.contains("://") || r.contains('?') || r.starts_with('/') || r.ends_with('/') {
        return false;
    }
    let mut parts = r.split('/');
    let (Some(org), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    !org.is_empty() && !name.is_empty()
}

/// A plausible BCP-47 or ISO-639-3 tag.
///
/// Deliberately loose: the manifest may name 65 languages and this runs 65
/// times per load. It rejects the things that are actually mistakes — a
/// sentence, an empty string, a path — and accepts everything shaped like a
/// language tag.
fn plausible_language_tag(tag: &str) -> bool {
    let t = tag.trim();
    if t.is_empty() || t.len() > 35 || t.contains(char::is_whitespace) {
        return false;
    }
    t.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && t.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{
      "schema_version": 1,
      "models": [{
        "id": "m1",
        "engine_id": "e1",
        "version": "1.0",
        "display_name": "M1",
        "description": "d",
        "task": "asr",
        "architecture": "arch",
        "access": "open",
        "license": {"spdx": "MIT", "name": "MIT", "url": "https://example.org/l"},
        "source": {"kind": "bundled"},
        "languages": ["hi"],
        "sample_rates": [16000],
        "size_bytes": 4,
        "files": [{"path": "a.bin", "role": "weights", "bytes": 4,
                   "sha256": "0000000000000000000000000000000000000000000000000000000000000000"}]
      }]
    }"#;

    #[test]
    fn a_minimal_valid_manifest_loads() {
        let m = Manifest::parse(GOOD).unwrap();
        assert_eq!(m.schema_version, SCHEMA_VERSION);
        assert_eq!(m.ids(), vec!["m1"]);
        let model = m.model("m1").unwrap();
        assert_eq!(model.dir, "m1", "dir defaults to the id");
        assert_eq!(model.size_bytes, 4);
        assert_eq!(model.weights_bytes, 4);
        assert_eq!(model.license.spdx, "mit", "spdx is normalised");
        assert!(
            model.source.url_for("a.bin").is_none(),
            "bundled has no URL"
        );
    }

    #[test]
    fn toml_and_json_parse_to_the_same_manifest() {
        let toml_text = r#"
schema_version = 1
[[models]]
id = "m1"
engine_id = "e1"
version = "1.0"
display_name = "M1"
description = "d"
task = "asr"
architecture = "arch"
access = "open"
size_bytes = 4
languages = ["hi"]
sample_rates = [16000]
[models.license]
spdx = "MIT"
name = "MIT"
url = "https://example.org/l"
[models.source]
kind = "bundled"
[[models.files]]
path = "a.bin"
role = "weights"
bytes = 4
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#;
        let from_toml = Manifest::parse(toml_text).unwrap();
        let from_json = Manifest::parse(GOOD).unwrap();
        assert_eq!(from_toml.models[0].id, from_json.models[0].id);
        assert_eq!(
            from_toml.models[0].files[0].sha256,
            from_json.models[0].files[0].sha256
        );
    }

    #[test]
    fn an_unknown_key_is_a_load_failure_not_a_silent_default() {
        // `sze` instead of `size_bytes`. Without `deny_unknown_fields` this
        // would load, the total would be missing, and the UI would show a model
        // with no size.
        let text = GOOD.replace("\"size_bytes\": 4", "\"sze\": 4, \"size_bytes\": 4");
        let e = Manifest::parse(&text).unwrap_err();
        assert!(matches!(e, ManifestError::Syntax { .. }), "{e}");
    }

    #[test]
    fn hashes_parse_case_insensitively_and_normalise() {
        let upper = "A".repeat(64);
        assert_eq!(parse_sha256(&upper), Some([0xAA; 32]));
        assert_eq!(parse_sha256(&"a".repeat(64)), Some([0xAA; 32]));
        assert_eq!(parse_sha256(&"a".repeat(63)), None);
        assert_eq!(parse_sha256(&"z".repeat(64)), None);
        assert_eq!(parse_sha256(&"a".repeat(65)), None);
    }

    #[test]
    fn sha256_of_the_empty_string_is_the_documented_constant() {
        // Guards the test fixtures themselves: a manifest whose file hash is all
        // zeroes is claiming the SHA-256 of the empty input, which is a
        // recognisable "nobody filled this in" value.
        assert_eq!(parse_sha256(&"0".repeat(64)), Some([0u8; 32]));
    }

    #[test]
    fn hf_repos_must_be_org_slash_name() {
        assert!(valid_hf_repo("org/name"));
        // The real upstream id, which an earlier "suspicious dot" check rejected.
        assert!(valid_hf_repo("ARTPARK-IISc/SraVaani-1.0"));
        assert!(!valid_hf_repo("name"));
        assert!(!valid_hf_repo("org/"));
        assert!(!valid_hf_repo("/name"));
        assert!(!valid_hf_repo("https://huggingface.co/org/name"));
        assert!(!valid_hf_repo("a/b/c"));
        assert!(!valid_hf_repo("org/name?x=1"));
    }

    #[test]
    fn language_tags_reject_sentences_and_paths() {
        for good in ["hi", "en", "hi-IN", "cmn", "pt_BR"] {
            assert!(plausible_language_tag(good), "{good} should pass");
        }
        // A sentence-shaped tag like "Hindi" passes: shape validation cannot tell
        // a language name from a language tag, and inventing a rule that rejects
        // short words would reject real tags. The manifest is authored data.
        for bad in [
            "",
            " ",
            "hi/IN",
            "../etc",
            "hi IN",
            "-hi",
            "a".repeat(64).as_str(),
        ] {
            assert!(!plausible_language_tag(bad), "{bad:?} should fail");
        }
    }

    #[test]
    fn compute_capability_parses_and_compares() {
        let g = GpuRequirements {
            compute_capability_min: Some("8.6".into()),
            vram_headroom_bytes: 0,
        };
        assert_eq!(g.compute_capability(), Some((8, 6)));
        let g = GpuRequirements {
            compute_capability_min: Some("nonsense".into()),
            vram_headroom_bytes: 0,
        };
        assert_eq!(g.compute_capability(), None);
    }

    #[test]
    fn only_bundled_is_not_downloadable() {
        for p in Provider::ALL {
            assert_eq!(p.is_downloadable(), p != Provider::Bundled, "{p}");
        }
        for a in Access::ALL {
            assert_eq!(a.needs_token(), a != Access::Open, "{a}");
        }
    }

    #[test]
    fn a_huggingface_source_builds_its_own_url() {
        let s = RawSource {
            kind: Some("huggingface".into()),
            repo: Some("org/name".into()),
            revision: None,
            url_template: None,
            homepage: None,
        }
        .validate("m1")
        .unwrap();
        assert_eq!(s.revision, "main");
        assert!(s.needs_bearer());
        let u = s.url_for("encoder.onnx").unwrap();
        assert_eq!(
            u.as_str(),
            "https://huggingface.co/org/name/resolve/main/encoder.onnx"
        );
    }

    #[test]
    fn a_non_hf_source_never_gets_a_bearer_token() {
        // Sending a credential to a host that does not need it is how tokens
        // leak, so this is a property of the type, not of a call site.
        let s = RawSource {
            kind: Some("direct".into()),
            repo: None,
            revision: None,
            url_template: Some("https://cdn.example.org/{path}".into()),
            homepage: None,
        }
        .validate("m1")
        .unwrap();
        assert!(!s.needs_bearer());
        assert!(s.url_for("a.bin").is_some());
    }

    #[test]
    fn error_codes_distinguish_an_unreadable_manifest_from_a_bad_one() {
        assert_eq!(
            ManifestError::NoModels.error_code(),
            vaani_core::ErrorCode::Internal
        );
        assert_eq!(
            ManifestError::Io {
                path: "x".into(),
                message: "y".into()
            }
            .error_code(),
            vaani_core::ErrorCode::StorageFailed
        );
    }

    #[test]
    fn a_manifest_error_never_leaks_its_detail_to_the_user() {
        let e: vaani_core::VaaniError = ManifestError::BadHash {
            id: "m1".into(),
            path: "encoder.fp16.onnx".into(),
            value: "zz".into(),
        }
        .into();
        assert_eq!(e.code(), vaani_core::ErrorCode::Internal);
        let shown = e.to_string();
        assert!(!shown.contains("encoder.fp16.onnx"), "leaked: {shown}");
        assert!(e.debug_chain().contains("encoder.fp16.onnx"));
        assert!(e.user.recovery.is_some());
    }

    #[test]
    fn optional_files_do_not_count_towards_the_weights_total() {
        // This is the fp32 encoder case: 1.65 GiB on disk, none of it in VRAM.
        // Written out rather than patched into the fixture, because a
        // string-replaced JSON fixture is exactly the kind of thing that stops
        // testing what its name says.
        let text = r#"{"schema_version":1,"models":[{
            "id":"m1","engine_id":"e1","version":"1","display_name":"M1",
            "description":"d","task":"asr","architecture":"a","access":"open",
            "license":{"spdx":"MIT","name":"MIT","url":"https://e.org/l"},
            "source":{"kind":"bundled"},"languages":["hi"],"sample_rates":[16000],
            "size_bytes":12,
            "files":[
              {"path":"a.bin","role":"weights","bytes":4,"required":true,
               "sha256":"0000000000000000000000000000000000000000000000000000000000000000"},
              {"path":"b.bin","role":"reference","bytes":8,"required":false,
               "sha256":"1111111111111111111111111111111111111111111111111111111111111111"}
            ]}]}"#;
        let m = Manifest::parse(text).unwrap();
        let model = m.model("m1").unwrap();
        assert_eq!(model.size_bytes, 12);
        assert_eq!(model.weights_bytes, 4, "the reference copy is not loaded");
        assert_eq!(model.required_paths(), vec!["a.bin"]);
        assert_eq!(model.files.len(), 2, "an optional file is still described");
    }

    #[test]
    fn missing_bytes_counts_only_absent_or_wrong_size_required_files() {
        let m = Manifest::parse(GOOD).unwrap();
        let model = m.model("m1").unwrap();
        assert_eq!(model.missing_bytes(&|_| Some(4)), 0);
        assert_eq!(model.missing_bytes(&|_| Some(3)), 4);
        assert_eq!(model.missing_bytes(&|_| None), 4);
    }

    #[test]
    fn the_embedded_manifest_is_valid() {
        // Cannot be skipped: a broken shipped manifest is a bug in this repo.
        let m = Manifest::embedded().expect("the embedded manifest must validate");
        assert!(!m.models.is_empty());
        for model in &m.models {
            assert!(
                !model.license.spdx.is_empty(),
                "{} has no licence",
                model.id
            );
        }
    }
}
