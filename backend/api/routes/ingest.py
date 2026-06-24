"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/api/routes/ingest.py

Document ingestion API endpoints.

Endpoints:
  POST /ingest/upload     — Upload and ingest a document
  GET  /ingest/documents  — List all ingested documents
  DELETE /ingest/document/{source_id} — Delete a document

The ingestion pipeline per request:
  1. Validate file (type, size)
  2. Save to disk temporarily
  3. Load + parse (DocumentLoader)
  4. Chunk semantically (DocumentChunker)
  5. Embed + store in ChromaDB (VectorStore)
  6. Update BM25 index (HybridSearchEngine)
  7. Return success with stats
"""

import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import (
    UnsupportedFileTypeError,
    FileTooLargeError,
    DocumentIngestionError,
    VectorStoreError,
)
from backend.rag.ingestion.loader import DocumentLoader
from backend.rag.ingestion.chunker import DocumentChunker, ChunkStrategy
from backend.rag.retrieval.vector_store import get_vector_store

settings = get_settings()
log = get_logger("ingest_route")

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Module-level singletons — initialised once, reused across requests
_loader = DocumentLoader()
_vector_store = get_vector_store()


# ══════════════════════════════════════════════════════════════════════
# Response Models
# ══════════════════════════════════════════════════════════════════════

class IngestResponse(BaseModel):
    """Response returned after successful document ingestion."""
    message: str
    source_id: str
    filename: str
    pages_loaded: int
    chunks_created: int
    chunk_strategy: str
    collection_stats: dict


class DocumentListResponse(BaseModel):
    """Response for listing all ingested documents."""
    sources: list[str]
    total_chunks: int
    collection_name: str


class DeleteResponse(BaseModel):
    """Response after document deletion."""
    message: str
    source_id: str


# ══════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════

@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Upload a PDF, TXT, MD, or DOCX file. "
        "PRISM will parse, chunk, embed, and store it in ChromaDB. "
        "Returns ingestion stats including chunk count and strategy used."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    chunk_strategy: ChunkStrategy = Form(
        default=ChunkStrategy.RECURSIVE,
        description="Chunking strategy: recursive | token | semantic",
    ),
    chunk_size: Optional[int] = Form(
        default=None,
        description="Override default chunk size (tokens/chars)",
    ),
    chunk_overlap: Optional[int] = Form(
        default=None,
        description="Override default chunk overlap",
    ),
    vision: bool = Form(
        default=False,
        description="Enable OCR extraction for images/charts in PDFs (Phase 2)",
    ),
) -> IngestResponse:
    """
    Full ingestion pipeline for a single document.

    Steps:
      1. Save uploaded file to data/uploads/
      2. Load and parse the file
      3. Chunk into semantic units
      4. Embed and store in ChromaDB
      5. Return stats

    The source_id returned can be used to delete this document later.
    """
    filename = file.filename or "unknown_file"
    source_id = str(uuid.uuid4())   # unique ID for this ingestion

    log.info(f"Ingestion request | file='{filename}' | strategy={chunk_strategy.value}")

    # ── Step 1: Save file to disk ─────────────────────────────────────
    upload_path = settings.upload_dir / f"{source_id}_{filename}"
    try:
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        log.info(f"File saved: {upload_path}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )
    finally:
        await file.close()

    # ── Steps 2-5: Run ingestion pipeline ────────────────────────────
    try:
        # Load
        documents = await _loader.load(
            file_path=upload_path,
            filename=filename,
            vision=vision,
        )

        if not documents:
            raise DocumentIngestionError(
                message=f"No content could be extracted from '{filename}'",
                detail="File may be empty, image-only PDF, or corrupted",
            )

        pages_loaded = len(documents)

        # Chunk
        chunker = DocumentChunker(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk(documents)

        # Store in ChromaDB
        _vector_store.add_documents(chunks, source_id=source_id)

        # Get updated stats
        stats = _vector_store.get_collection_stats()

        log.info(
            f"Ingestion complete | "
            f"file='{filename}' | "
            f"pages={pages_loaded} | "
            f"chunks={len(chunks)} | "
            f"source_id={source_id}"
        )

        return IngestResponse(
            message=f"Successfully ingested '{filename}'",
            source_id=source_id,
            filename=filename,
            pages_loaded=pages_loaded,
            chunks_created=len(chunks),
            chunk_strategy=chunk_strategy.value,
            collection_stats=stats,
        )

    except (UnsupportedFileTypeError, FileTooLargeError, DocumentIngestionError):
        # Clean up the saved file before re-raising
        _cleanup_file(upload_path)
        raise

    except VectorStoreError:
        _cleanup_file(upload_path)
        raise

    except Exception as e:
        _cleanup_file(upload_path)
        log.exception(f"Unexpected error during ingestion of '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
    description="Returns all document sources currently stored in ChromaDB.",
)
async def list_documents() -> DocumentListResponse:
    """
    List all documents currently in the vector store.

    Used by the Streamlit sidebar to show which documents
    are available for querying.
    """
    try:
        sources = _vector_store.list_sources()
        stats = _vector_store.get_collection_stats()

        return DocumentListResponse(
            sources=sources,
            total_chunks=stats["chunk_count"],
            collection_name=stats["collection_name"],
        )
    except VectorStoreError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )


@router.delete(
    "/document/{source_id}",
    response_model=DeleteResponse,
    summary="Delete an ingested document",
    description="Remove all chunks for a document from ChromaDB using its source_id.",
)
async def delete_document(source_id: str) -> DeleteResponse:
    """
    Delete all chunks belonging to a specific document.

    The source_id is returned during ingestion.
    Useful for re-ingesting an updated version of a document.
    """
    log.info(f"Delete request | source_id='{source_id}'")

    try:
        _vector_store.delete_document(source_id=source_id)
        return DeleteResponse(
            message=f"Document '{source_id}' deleted successfully",
            source_id=source_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _cleanup_file(path: Path) -> None:
    """Silently delete a file — used for cleanup after failed ingestion."""
    try:
        if path.exists():
            path.unlink()
            log.debug(f"Cleaned up temp file: {path}")
    except Exception:
        pass