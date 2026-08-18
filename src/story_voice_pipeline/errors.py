"""Domain errors that are safe to present to CLI users."""


class PipelineError(RuntimeError):
    """Base error for expected pipeline failures."""


class LibraryValidationError(PipelineError):
    """The story library is invalid."""


class AudioValidationError(PipelineError):
    """Audio could not be read or failed a required validation."""


class ResumeError(PipelineError):
    """A run cannot be resumed safely."""


class EngineError(PipelineError):
    """The selected TTS engine failed."""
