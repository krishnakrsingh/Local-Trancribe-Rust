//! Pipeline configuration: model, device, language, diarization, work dir.
//!
//! [`AppConfig`] is the orchestrator's view of the settings. The UI speaks
//! [`AppSettings`](vaani_ui::backend::AppSettings); the backend translates at
//! the boundary with [`AppConfig::from_settings`] so the pipeline never
//! imports UI strings.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vaani_core::Language;
use vaani_inference::Provider;
use vaani_ui::backend::{AppSettings, Theme};

/// The default ASR model: the manifest id in `vaani-models`' embedded
/// registry.
pub const DEFAULT_MODEL: &str = "sravaani-1.0-onnx";

/// How many times a transient decode failure is retried, by default.
pub const DEFAULT_DECODE_RETRIES: u32 = 1;

/// Everything one pipeline run needs to know that is not the input file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppConfig {
    /// Manifest id of the ASR model, e.g. `sravaani-1.0-onnx`.
    pub model: String,
    /// Compute device wire form: `auto`, `cpu` or `cuda`.
    pub device: String,
    /// Language hint handed to the ASR engine per chunk.
    pub language: Language,
    /// Whether to run speaker separation. `false` skips diarization with a
    /// reason instead of failing the job.
    pub diarize: bool,
    /// Parent directory for per-job work dirs (`work_dir/<job-id>/`).
    pub work_dir: PathBuf,
    /// Retries for transient decode failures, on top of the first attempt.
    pub decode_retries: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_owned(),
            device: "auto".to_owned(),
            language: Language::Auto,
            diarize: true,
            work_dir: default_work_dir(),
            decode_retries: DEFAULT_DECODE_RETRIES,
        }
    }
}

impl AppConfig {
    /// The inference provider named by [`AppConfig::device`].
    ///
    /// Unknown values fall back to `Auto`: a typo in a settings file must not
    /// refuse to transcribe.
    #[must_use]
    pub fn provider(&self) -> Provider {
        Provider::parse(&self.device).unwrap_or(Provider::Auto)
    }

    /// Total decode attempts: the first try plus
    /// [`AppConfig::decode_retries`]. Always at least one.
    #[must_use]
    pub fn max_decode_attempts(&self) -> u32 {
        self.decode_retries.saturating_add(1).max(1)
    }

    /// Translate the UI's settings into a pipeline config.
    ///
    /// The UI's language is a BCP-47-ish tag with the empty string meaning
    /// "detect"; see [`language_from_setting`].
    #[must_use]
    pub fn from_settings(settings: &AppSettings) -> Self {
        Self {
            model: settings.model.clone(),
            device: settings.device.clone(),
            language: language_from_setting(&settings.language),
            diarize: settings.diarize,
            work_dir: default_work_dir(),
            decode_retries: DEFAULT_DECODE_RETRIES,
        }
    }
}

/// Where per-job work dirs live when nothing else is configured.
///
/// `%LOCALAPPDATA%/VaaniDesk/jobs` on Windows via [`dirs::data_dir`], falling
/// back to the system temp dir when the platform has no data dir.
#[must_use]
pub fn default_work_dir() -> PathBuf {
    if let Some(base) = dirs::data_dir() {
        base.join("VaaniDesk").join("jobs")
    } else {
        std::env::temp_dir().join("vaanidesk-jobs")
    }
}

/// Map the UI's language tag to the domain [`Language`].
///
/// The settings contract says the empty string means "detect". Unknown tags
/// also map to `Auto` rather than failing: the model decides per segment.
#[must_use]
pub fn language_from_setting(tag: &str) -> Language {
    let trimmed = tag.trim();
    if trimmed.is_empty() {
        return Language::Auto;
    }
    Language::parse(trimmed).unwrap_or(Language::Auto)
}

/// The settings the backend starts with before anything is persisted.
#[must_use]
pub fn default_app_settings() -> AppSettings {
    AppSettings {
        model: DEFAULT_MODEL.to_owned(),
        device: "auto".to_owned(),
        language: String::new(),
        diarize: true,
        theme: Theme::System,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_select_the_bundled_model_and_auto_device() {
        let config = AppConfig::default();
        assert_eq!(config.model, "sravaani-1.0-onnx");
        assert_eq!(config.provider(), Provider::Auto);
        assert_eq!(config.language, Language::Auto);
        assert!(config.diarize);
        assert_eq!(config.max_decode_attempts(), 2);
    }

    #[test]
    fn unknown_device_falls_back_to_auto() {
        let config = AppConfig {
            device: "quantum".to_owned(),
            ..AppConfig::default()
        };
        assert_eq!(config.provider(), Provider::Auto);
        for (tag, provider) in [
            ("cpu", Provider::Cpu),
            ("cuda", Provider::RequireCuda),
            ("auto", Provider::Auto),
        ] {
            let config = AppConfig {
                device: tag.to_owned(),
                ..AppConfig::default()
            };
            assert_eq!(config.provider(), provider, "{tag}");
        }
    }

    #[test]
    fn settings_translation_maps_detect_and_tags() {
        let mut settings = default_app_settings();
        assert_eq!(AppConfig::from_settings(&settings).language, Language::Auto);
        settings.language = "hi".to_owned();
        assert_eq!(
            AppConfig::from_settings(&settings).language,
            Language::Hindi
        );
        settings.language = "klingon".to_owned();
        assert_eq!(AppConfig::from_settings(&settings).language, Language::Auto);
    }

    #[test]
    fn retry_budget_is_saturating_and_at_least_one() {
        let config = AppConfig {
            decode_retries: u32::MAX,
            ..AppConfig::default()
        };
        assert_eq!(config.max_decode_attempts(), u32::MAX);
    }
}
