"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/core/exceptions.py

Custom exception hierarchy for PRISM.

Every error in the application raises one of these typed exceptions.
FastAPI's exception handlers (registered in main.py) catch them and
return clean, structured JSON responses instead of raw Python tracebacks.

Design pattern: each exception carries:
  - A human-readable message
  - An optional detail field for debugging context
  - A fixed HTTP status code (as a class attribute)

Usage:
    from backend.core.exceptions import DocumentNotFoundError

    raise DocumentNotFoundError(
        message="No document found with that ID",
        detail=f"searched collection: {collection_name}"
    )
"""

from typing import Any


# ══════════════════════════════════════════════════════════════════════
# Base Exception
# ══════════════════════════════════════════════════════════════════════

class PRISMBaseException(Exception):
    """
    Root exception for all PRISM errors.

    All custom exceptions inherit from this so callers can catch
    either a specific error or any PRISM error with one except clause:

        except PRISMBaseException as e:
            logger.error(e.message)
    """

    http_status: int = 500
    error_code: str = "PRISM_ERROR"

    def __init__(
        self,
        message: str = "An unexpected error occurred in PRISM",
        detail: Any = None,
    ) -> None:
        self.message = message
        self.detail = detail
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Serialise to a dict for JSON API responses."""
        payload = {
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.detail is not None:
            payload["detail"] = str(self.detail)
        return payload

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, detail={self.detail!r})"


# ══════════════════════════════════════════════════════════════════════
# Configuration Errors
# ══════════════════════════════════════════════════════════════════════

class ConfigurationError(PRISMBaseException):
    """
    Raised when a required setting is missing or invalid at startup.
    Causes the application to refuse to start rather than run misconfigured.

    Example: OPENAI_API_KEY is empty when llm_provider == 'openai'
    """
    http_status = 500
    error_code = "CONFIGURATION_ERROR"

    def __init__(self, message: str = "Invalid or missing configuration", detail: Any = None) -> None:
        super().__init__(message, detail)


# ══════════════════════════════════════════════════════════════════════
# Document & Ingestion Errors
# ══════════════════════════════════════════════════════════════════════

class DocumentIngestionError(PRISMBaseException):
    """
    Raised when a document fails during the ingestion pipeline.
    Covers: loading, parsing, chunking, and embedding failures.

    Example: a corrupted PDF that pypdf cannot parse
    """
    http_status = 422
    error_code = "DOCUMENT_INGESTION_ERROR"

    def __init__(self, message: str = "Failed to ingest document", detail: Any = None) -> None:
        super().__init__(message, detail)


class DocumentNotFoundError(PRISMBaseException):
    """
    Raised when a requested document ID does not exist in ChromaDB.

    Example: user queries a document that was never uploaded
    """
    http_status = 404
    error_code = "DOCUMENT_NOT_FOUND"

    def __init__(self, message: str = "Document not found", detail: Any = None) -> None:
        super().__init__(message, detail)


class UnsupportedFileTypeError(PRISMBaseException):
    """
    Raised when an uploaded file's extension is not in the allowed list.

    Example: user uploads a .exe or .zip file
    """
    http_status = 415
    error_code = "UNSUPPORTED_FILE_TYPE"

    def __init__(self, message: str = "File type not supported", detail: Any = None) -> None:
        super().__init__(message, detail)


class FileTooLargeError(PRISMBaseException):
    """
    Raised when an uploaded file exceeds MAX_UPLOAD_SIZE_MB.

    Example: user uploads a 200MB PDF when limit is 50MB
    """
    http_status = 413
    error_code = "FILE_TOO_LARGE"

    def __init__(self, message: str = "File size exceeds the allowed limit", detail: Any = None) -> None:
        super().__init__(message, detail)


# ══════════════════════════════════════════════════════════════════════
# Retrieval Errors
# ══════════════════════════════════════════════════════════════════════

class RetrievalError(PRISMBaseException):
    """
    Raised when the hybrid search or vector store query fails.

    Example: ChromaDB collection is empty or corrupted
    """
    http_status = 500
    error_code = "RETRIEVAL_ERROR"

    def __init__(self, message: str = "Failed to retrieve relevant documents", detail: Any = None) -> None:
        super().__init__(message, detail)


class EmbeddingError(PRISMBaseException):
    """
    Raised when the embedding model fails to encode text.

    Example: sentence-transformers model not downloaded yet,
             or input text exceeds model's max token length
    """
    http_status = 500
    error_code = "EMBEDDING_ERROR"

    def __init__(self, message: str = "Failed to generate text embeddings", detail: Any = None) -> None:
        super().__init__(message, detail)


class VectorStoreError(PRISMBaseException):
    """
    Raised when ChromaDB operations fail (read, write, delete).

    Example: disk full, permission denied on chroma_persist_dir
    """
    http_status = 500
    error_code = "VECTOR_STORE_ERROR"

    def __init__(self, message: str = "Vector store operation failed", detail: Any = None) -> None:
        super().__init__(message, detail)


# ══════════════════════════════════════════════════════════════════════
# Generation Errors
# ══════════════════════════════════════════════════════════════════════

class LLMError(PRISMBaseException):
    """
    Raised when the LLM API call fails.

    Example: OpenAI rate limit hit, invalid API key,
             model context window exceeded
    """
    http_status = 502
    error_code = "LLM_ERROR"

    def __init__(self, message: str = "LLM generation failed", detail: Any = None) -> None:
        super().__init__(message, detail)


class ContextWindowExceededError(PRISMBaseException):
    """
    Raised when the retrieved chunks + query exceed the LLM's token limit.
    Caller should reduce retrieval_top_k or chunk_size and retry.
    """
    http_status = 422
    error_code = "CONTEXT_WINDOW_EXCEEDED"

    def __init__(self, message: str = "Input exceeds LLM context window", detail: Any = None) -> None:
        super().__init__(message, detail)


# ══════════════════════════════════════════════════════════════════════
# Vision / OCR Errors  (Phase 2)
# ══════════════════════════════════════════════════════════════════════

class VisionProcessingError(PRISMBaseException):
    """
    Raised when OpenCV or EasyOCR fails to process an image or chart.

    Example: image is too blurry for OCR, unsupported image format
    """
    http_status = 422
    error_code = "VISION_PROCESSING_ERROR"

    def __init__(self, message: str = "Failed to process image or chart", detail: Any = None) -> None:
        super().__init__(message, detail)


# ══════════════════════════════════════════════════════════════════════
# Evaluation Errors
# ══════════════════════════════════════════════════════════════════════

class EvaluationError(PRISMBaseException):
    """
    Raised when Ragas evaluation pipeline fails.

    Example: no ground truth provided, dataset format mismatch
    """
    http_status = 500
    error_code = "EVALUATION_ERROR"

    def __init__(self, message: str = "RAG evaluation pipeline failed", detail: Any = None) -> None:
        super().__init__(message, detail)


# ══════════════════════════════════════════════════════════════════════
# API / Request Errors
# ══════════════════════════════════════════════════════════════════════

class InvalidRequestError(PRISMBaseException):
    """
    Raised when the request payload is structurally valid JSON
    but semantically invalid for the operation.

    Example: query string is empty, top_k is negative
    """
    http_status = 400
    error_code = "INVALID_REQUEST"

    def __init__(self, message: str = "Invalid request parameters", detail: Any = None) -> None:
        super().__init__(message, detail)


class RateLimitError(PRISMBaseException):
    """
    Raised when upstream API rate limits are hit (OpenAI, HuggingFace).
    The retry decorator in generation/chain.py handles this automatically,
    but if all retries are exhausted this exception surfaces to the user.
    """
    http_status = 429
    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(self, message: str = "Rate limit exceeded, please try again later", detail: Any = None) -> None:
        super().__init__(message, detail)
