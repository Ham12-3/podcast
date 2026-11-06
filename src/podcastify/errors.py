"""Errors with messages meant for the user (the CLI and UI show them without a traceback)."""


class PodcastifyError(RuntimeError):
    """Base class for problems the user can understand and act on."""


class ExtractionError(PodcastifyError):
    """Could not get usable text from the input (bad URL, empty PDF, no captions...)."""


class LLMError(PodcastifyError):
    """Ollama is unreachable, the model is missing, or it returned an unusable script."""


class AudioError(PodcastifyError):
    """Problems stitching or exporting audio (for example ffmpeg missing)."""
