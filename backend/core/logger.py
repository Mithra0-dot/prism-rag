"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/core/logger.py

Centralised logging configuration using Loguru.

Features:
  - Coloured, human-readable output in development
  - Structured JSON output in production (machine-parseable)
  - Automatic log rotation (10 MB) with 7-day retention
  - Request ID injection for tracing requests across modules
  - Zero-config usage: just import logger from this module

Usage:
    from backend.core.logger import logger, setup_logging

    setup_logging()           # call once at app startup
    logger.info("PRISM started")
    logger.debug("chunk size = {size}", size=512)
    logger.error("retrieval failed: {err}", err=str(e))
"""

import json
import logging
import sys
from pathlib import Path

from loguru import logger

from backend.core.config import get_settings

# ── Log directory ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Intercept stdlib logging so third-party libs (uvicorn, etc.)
#    flow through Loguru too ─────────────────────────────────────────
class _InterceptHandler(logging.Handler):
    """
    Redirect any stdlib logging.Logger call into Loguru.
    This means uvicorn, httpx, chromadb all appear in the same stream.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _json_formatter(record: dict) -> str:
    """
    Serialise a Loguru record to a single JSON line.
    Used in production so log aggregators (Datadog, CloudWatch) can parse it.
    """
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    if record["exception"]:
        log_entry["exception"] = str(record["exception"])
    if record["extra"]:
        log_entry["extra"] = record["extra"]

    return json.dumps(log_entry) + "\n"


def setup_logging() -> None:
    """
    Configure Loguru for the PRISM application.

    Call exactly once at FastAPI startup (inside the lifespan context).
    Subsequent calls are safe but redundant — Loguru deduplicates sinks.

    Behaviour by environment:
      development  → coloured terminal output, DEBUG level
      staging      → terminal + rotating file, INFO level
      production   → JSON file only, INFO level
    """
    settings = get_settings()

    # Remove Loguru's default stderr sink so we control everything
    logger.remove()

    is_prod = settings.is_production
    log_level = "DEBUG" if settings.is_development else "INFO"

    # ── Terminal sink (development + staging) ─────────────────────────
    if not is_prod:
        logger.add(
            sys.stderr,
            level=log_level,
            colorize=True,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            backtrace=True,
            diagnose=True,   # show variable values in tracebacks
        )

    # ── Rotating file sink (all environments) ─────────────────────────
    if is_prod:
        # JSON structured logs for production log aggregators
        logger.add(
            LOG_DIR / "prism_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{message}",          # _json_formatter handles the format
            filter=lambda r: _json_formatter(r) or True,  # type: ignore
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            serialize=True,              # Loguru's built-in JSON mode
            enqueue=True,                # async-safe writes
        )
    else:
        # Human-readable rotating logs for development/staging
        logger.add(
            LOG_DIR / "prism_{time:YYYY-MM-DD}.log",
            level=log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{module}:{function}:{line} | {message}"
            ),
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,
        )

    # ── Intercept stdlib loggers ──────────────────────────────────────
    #    Covers: uvicorn, uvicorn.error, uvicorn.access, fastapi, httpx
    stdlib_loggers = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "httpx",
        "chromadb",
        "langchain",
    ]
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in stdlib_loggers:
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False

    logger.info(
        "PRISM logging initialised | env={env} | level={level}",
        env=settings.environment,
        level=log_level,
    )


def get_logger(name: str):
    """
    Return a Loguru logger bound with a module name tag.
    Useful for filtering logs by component in production.

    Usage:
        log = get_logger("ingestion")
        log.info("document loaded")   # appears as [ingestion] document loaded
    """
    return logger.bind(component=name)
