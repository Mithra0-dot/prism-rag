"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/main.py

FastAPI application entry point.

Responsibilities:
  - Create and configure the FastAPI app instance
  - Register global exception handlers (maps custom exceptions → HTTP responses)
  - Configure CORS so the Streamlit frontend can communicate with this API
  - Lifespan context manager for startup/shutdown logic
  - Mount all API routers under /api/v1
  - Health check endpoint

Run the server:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.logger import get_logger, setup_logging
from backend.core.exceptions import (
    PRISMBaseException,
    DocumentNotFoundError,
    UnsupportedFileTypeError,
    FileTooLargeError,
    LLMError,
    RateLimitError,
)
from backend.api.routes.ingest import router as ingest_router
from backend.api.routes.query import router as query_router

settings = get_settings()
log = get_logger("main")


# ══════════════════════════════════════════════════════════════════════
# Lifespan — startup & shutdown
# ══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager replaces the old @app.on_event pattern.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.

    This is the recommended FastAPI pattern as of v0.93+
    """
    # ── Startup ───────────────────────────────────────────────────────
    setup_logging()
    log.info("=" * 60)
    log.info("PRISM is starting up...")
    log.info(f"Version      : {settings.app_version}")
    log.info(f"Environment  : {settings.environment}")
    log.info(f"LLM Provider : {settings.llm_provider}")
    log.info(f"LLM Model    : {settings.llm_model_name}")
    log.info(f"Embedding    : {settings.embedding_model}")
    log.info(f"ChromaDB     : {settings.chroma_persist_dir}")
    log.info(f"API Prefix   : {settings.api_prefix}")
    log.info("=" * 60)

    # Ensure required data directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    log.info("Data directories verified")

    # Validate critical configuration early
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        log.warning(
            "OPENAI_API_KEY is not set. "
            "LLM generation will fail until a key is provided."
        )

    log.info("PRISM startup complete. Ready to serve requests.")

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────
    log.info("PRISM is shutting down. Cleaning up resources...")
    log.info("Shutdown complete. Goodbye.")


# ══════════════════════════════════════════════════════════════════════
# App Instance
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url=f"{settings.api_prefix}/docs",       # Swagger UI
    redoc_url=f"{settings.api_prefix}/redoc",     # ReDoc UI
    openapi_url=f"{settings.api_prefix}/openapi.json",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════
# Middleware
# ══════════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # ["http://localhost:8501"] by default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
# Exception Handlers
# ══════════════════════════════════════════════════════════════════════
# Each handler catches a specific exception type and returns a structured
# JSON response. The HTTP status code comes from the exception class itself.

def _error_response(exc: PRISMBaseException) -> JSONResponse:
    """Helper — builds a consistent JSON error payload from any PRISM exception."""
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
    )


@app.exception_handler(PRISMBaseException)
async def prism_base_exception_handler(
    request: Request, exc: PRISMBaseException
) -> JSONResponse:
    log.error(f"[{exc.error_code}] {exc.message} | detail={exc.detail}")
    return _error_response(exc)


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(
    request: Request, exc: DocumentNotFoundError
) -> JSONResponse:
    log.warning(f"Document not found: {exc.message}")
    return _error_response(exc)


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_type_handler(
    request: Request, exc: UnsupportedFileTypeError
) -> JSONResponse:
    log.warning(f"Unsupported file type: {exc.detail}")
    return _error_response(exc)


@app.exception_handler(FileTooLargeError)
async def file_too_large_handler(
    request: Request, exc: FileTooLargeError
) -> JSONResponse:
    log.warning(f"File too large: {exc.detail}")
    return _error_response(exc)


@app.exception_handler(LLMError)
async def llm_error_handler(
    request: Request, exc: LLMError
) -> JSONResponse:
    log.error(f"LLM error: {exc.message} | {exc.detail}")
    return _error_response(exc)


@app.exception_handler(RateLimitError)
async def rate_limit_handler(
    request: Request, exc: RateLimitError
) -> JSONResponse:
    log.warning("Rate limit hit — advising client to retry")
    return _error_response(exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all for any exception we didn't anticipate.
    Logs the full traceback but returns a safe generic message to the client.
    Never expose raw Python tracebacks in API responses.
    """
    log.exception(f"Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# ══════════════════════════════════════════════════════════════════════
# Core Routes
# ══════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root() -> dict:
    """Redirect hint for anyone hitting the bare root URL."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_prefix}/docs",
        "health": f"{settings.api_prefix}/health",
    }


@app.get(
    f"{settings.api_prefix}/health",
    tags=["System"],
    summary="Health check",
    response_description="Returns service status and basic config info",
)
async def health_check() -> dict:
    """
    Health check endpoint.

    Used by:
      - Docker HEALTHCHECK instruction
      - GitHub Actions CI pipeline
      - Streamlit frontend (to verify backend is reachable before querying)

    Returns 200 if the service is running correctly.
    """
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model,
    }


# ══════════════════════════════════════════════════════════════════════
# API Routers
# ══════════════════════════════════════════════════════════════════════


app.include_router(ingest_router, prefix=settings.api_prefix)
app.include_router(query_router, prefix=settings.api_prefix)


# ══════════════════════════════════════════════════════════════════════
# Dev entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level="debug" if settings.is_development else "info",
    )
