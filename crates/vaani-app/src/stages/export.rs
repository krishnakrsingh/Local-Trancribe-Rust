//! The export stage: render the transcript to a file on demand.
//!
//! Runs at the end of the pipeline order (`INTEGRATION_CONTRACT.md` §14) but
//! only when asked: the orchestrator invokes it from `export_job`, never as
//! part of transcription. One stage per export request, holding its format,
//! target and options.

use vaani_core::pipeline::{PipelineStage, StageContext};
use vaani_core::{JobPhase, Result};
use vaani_export::{ExportFormat, ExportOptions, ExportTarget};

/// Render the accumulated transcript to one file.
#[derive(Debug)]
pub struct ExportStage {
    format: ExportFormat,
    target: ExportTarget,
    options: ExportOptions,
}

impl ExportStage {
    /// A stage exporting `format` to `target` with `options`.
    pub fn new(format: ExportFormat, target: ExportTarget, options: ExportOptions) -> Self {
        Self {
            format,
            target,
            options,
        }
    }

    /// The requested format.
    #[must_use]
    pub fn format(&self) -> ExportFormat {
        self.format
    }
}

impl PipelineStage for ExportStage {
    fn name(&self) -> &'static str {
        "export"
    }

    fn phase(&self) -> JobPhase {
        JobPhase::Exporting
    }

    fn run(&self, ctx: &mut StageContext<'_>) -> Result<()> {
        ctx.check()?;
        let transcript = ctx.artifacts.transcript()?;
        let exported =
            vaani_export::export_to_file(transcript, self.format, &self.target, &self.options)?;
        ctx.report(
            JobPhase::Exporting,
            Some(1.0),
            Some(&format!("wrote {}", exported.path.display())),
        );
        ctx.artifacts.exported.push(exported.path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::segment;
    use std::path::Path;
    use vaani_core::pipeline::StageArtifacts;
    use vaani_core::{CancelToken, EngineStamp, Time, Transcript};

    #[test]
    fn export_stage_writes_and_records_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let target = ExportTarget::Directory(dir.path().to_path_buf());
        let options = ExportOptions::default().with_source_name("meeting.wav");
        let stage = ExportStage::new(ExportFormat::Text, target, options);
        assert_eq!(stage.name(), "export");
        assert_eq!(stage.phase(), JobPhase::Exporting);

        let mut transcript = Transcript::new(EngineStamp::asr("fake-asr", "test.0"));
        transcript.push_chunk(Time::ZERO, vec![segment("hello world", 0, 1000)]);
        let mut artifacts = StageArtifacts {
            transcript: Some(transcript),
            ..StageArtifacts::default()
        };
        let cancel = CancelToken::new();
        let mut ctx = StageContext {
            input: Path::new("meeting.wav"),
            work_dir: dir.path(),
            cancel: &cancel,
            progress: &|_| {},
            artifacts: &mut artifacts,
        };
        stage.run(&mut ctx).unwrap();
        assert_eq!(ctx.artifacts.exported.len(), 1);
        assert!(ctx.artifacts.exported[0].exists());
    }
}
