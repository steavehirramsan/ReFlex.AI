"""Exception hierarchy for ReFlex.

All recoverable, ReFlex-specific failures derive from :class:`ReflexError` so callers
can catch the whole family without swallowing unrelated exceptions.
"""

from __future__ import annotations


class ReflexError(Exception):
    """Base class for all ReFlex errors."""


class ConfigError(ReflexError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class MemoryError_(ReflexError):
    """Raised when a memory tier cannot satisfy a read/write operation.

    Named with a trailing underscore to avoid shadowing the built-in
    :class:`MemoryError`; exported as ``MemoryError_`` and re-exported as
    ``ReflexMemoryError``.
    """


ReflexMemoryError = MemoryError_


class BackendError(ReflexError):
    """Raised when a pluggable backend (LLM, embeddings, vector index) fails."""


class LLMError(BackendError):
    """Raised when the language-model backend fails or returns an invalid response."""


class IntegrityViolation(ReflexError):
    """Raised when content fails an integrity check that is configured to be fatal.

    By default the integrity layer *flags* rather than raises; this is used only
    when ``integrity.on_violation == "raise"``.
    """

    def __init__(self, message: str, *, flags: list[str] | None = None) -> None:
        super().__init__(message)
        self.flags = flags or []
