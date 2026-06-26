"""Structured logging for the txt2reward package."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO, *, stream=None) -> None:
    """
    Configure the ``txt2reward`` logger tree.

    Uses a message-only formatter so CLI output matches the legacy ``print``
    lines (e.g. ``[train] ...``). Safe to call multiple times; updates level.
    """
    global _CONFIGURED
    logger = logging.getLogger("txt2reward")
    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    _CONFIGURED = True


def _ensure_configured() -> None:
    if not _CONFIGURED:
        configure_logging()


def get_logger(component: str) -> logging.Logger:
    """Return a component logger under the ``txt2reward`` namespace."""
    _ensure_configured()
    return logging.getLogger(f"txt2reward.{component}")
