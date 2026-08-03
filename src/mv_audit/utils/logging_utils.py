"""Logging helpers for command-line scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure console logging and an optional file handler."""

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger without changing global configuration."""

    return logging.getLogger(name)
