"""Logging setup for ReFlex.

A single :func:`get_logger` entry point returns namespaced loggers under ``reflex.*``.
:func:`configure_logging` installs a Rich handler when available (pretty, leveled console
output) and falls back to the stdlib formatter otherwise, so importing the library never
forces a logging side effect on the host application.
"""

from __future__ import annotations

import logging
from typing import Literal

_CONFIGURED = False

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(level: LogLevel | str | int = "INFO", *, use_rich: bool = True) -> None:
    """Configure the root ``reflex`` logger. Idempotent.

    Parameters
    ----------
    level:
        Logging level name or numeric level.
    use_rich:
        If True and ``rich`` is installed, use ``RichHandler`` for colourised output.
    """
    global _CONFIGURED
    logger = logging.getLogger("reflex")
    logger.setLevel(level)
    logger.handlers.clear()

    handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:  # pragma: no cover - rich is a core dep but stay defensive
            handler = _plain_handler()
    else:
        handler = _plain_handler()

    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def _plain_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring sane defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    suffix = name.split("reflex.", 1)[-1]
    return logging.getLogger(f"reflex.{suffix}" if not name.startswith("reflex") else name)
