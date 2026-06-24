"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/api/routes/query.py

RAG query API endpoints.

Endpoints:
  POST /query/ask         — One-shot Q&A (returns full answer)
  POST /query/stream      — Streaming Q&A (SSE token stream)
  POST /query/search      — Raw retrieval only (no generation)
  GET  /query/sources     — Sources from last query

The streaming endpoint is what the Streamlit frontend uses.
Server-Sent Events (SSE) allow the server to push tokens to the
client as they're generated — the "typing" effect users see.
"""

import json
import asyncio
from typing import Optional, AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import (
    LLMError,
    RetrievalError,
    RateLimitError,
    VectorStoreError,
)
from backend.rag.retrieval.hybrid_search import HybridSearchEngine
from backend.rag.retrieval.vector_store import get_vector_store
from backend.rag.generation.chain import PRISMChain
from backend.rag.generation.prompt import PromptType

settings = get_settings()
log = get_logger("query_route")

router = APIRouter(prefix="/query", tags=["Query"])

# Module-level singletons
_vector_store = get_vector_store()
_search_engine = HybridSearchEngine()
_chain = PRISMChain(search_engine=_search_engine)


# ══════════════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Request body for Q&A endpoints."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about your documents",
        examples=["What was the revenue in Q3?"],
    )
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (default: from config)",
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Limit search to a specific document filename",
    )
    prompt_type: PromptType = Field(
        default=PromptType.QA,
        description="Prompt style: qa | conversational | summary",
    )


class SearchRequest(BaseModel):
    """Request body for raw retrieval without generation."""
    query: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    source_filter: Optional[str] = None


class QueryResponse(BaseModel):
    """Response for non-streaming Q&A."""
    answer: str
    sources: list[dict]
    chunks_used: int
    latency_seconds: float


class SearchResponse(BaseModel):
    """Response for raw retrieval endpoint."""
    results: list[dict]
    total_found: int
    query: str


# ══════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════

@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question (full response)",
    description=(
        "Ask a question about your ingested documents. "
        "Returns the complete answer at once. "
        "For real-time streaming use /query/stream instead."
    ),
)
async def ask(request: QueryRequest) -> QueryResponse:
    """
    Non-streaming RAG Q&A endpoint.

    Useful for:
      - Programmatic API usage where streaming is not needed
      - Ragas evaluation pipeline (needs full answer at once)
      - Testing and debugging

    Returns the full answer, sources, and latency in one response.
    """
    log.info(f"Query request | question='{request.question[:80]}'")

    # Build metadata filter if source is specified
    metadata_filter = None
    if request.source_filter:
        metadata_filter = {"source": request.source_filter}

    try:
        chain = PRISMChain(
            prompt_type=request.prompt_type,
            search_engine=_search_engine,
        )
        result = chain.query(
            question=request.question,
            k=request.k,
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=len(result["chunks"]),
            latency_seconds=result["latency"],
        )

    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.message,
        )
    except LLMError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=e.message,
        )


@router.post(
    "/stream",
    summary="Ask a question (streaming response)",
    description=(
        "Ask a question and receive the answer as a Server-Sent Events stream. "
        "Tokens arrive as they are generated — enables the real-time typing effect. "
        "This is the primary endpoint used by the Streamlit frontend."
    ),
    response_class=StreamingResponse,
)
async def stream(request: QueryRequest) -> StreamingResponse:
    """
    Streaming RAG Q&A via Server-Sent Events (SSE).

    SSE format — each event looks like:
        data: {"type": "token", "content": "The "}\n\n
        data: {"type": "token", "content": "revenue"}\n\n
        data: {"type": "sources", "content": [...]}\n\n
        data: {"type": "done", "content": ""}\n\n

    The Streamlit frontend reads these events and appends each
    token to the displayed text in real time.

    Why SSE over WebSockets:
      SSE is simpler, works over standard HTTP, and is sufficient
      for one-directional server→client streaming. WebSockets would
      be overkill here and harder to deploy behind a proxy.
    """
    log.info(f"Stream request | question='{request.question[:80]}'")

    metadata_filter = None
    if request.source_filter:
        metadata_filter = {"source": request.source_filter}

    async def event_generator() -> AsyncIterator[str]:
        """
        Generate SSE events for each token and then sources.

        The finally block ensures we always send a 'done' event
        so the client knows the stream has ended, even on error.
        """
        chain = PRISMChain(
            prompt_type=request.prompt_type,
            search_engine=_search_engine,
        )

        try:
            # Stream tokens
            async for token in chain.astream(
                question=request.question,
                k=request.k,
                filter=metadata_filter,
            ):
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"

            # After all tokens — send sources
            sources = chain.get_sources()
            sources_event = json.dumps({"type": "sources", "content": sources})
            yield f"data: {sources_event}\n\n"

        except RetrievalError as e:
            error_event = json.dumps({
                "type": "error",
                "content": f"Retrieval failed: {e.message}",
            })
            yield f"data: {error_event}\n\n"
            log.error(f"Retrieval error during stream: {e.message}")

        except RateLimitError as e:
            error_event = json.dumps({
                "type": "error",
                "content": e.message,
            })
            yield f"data: {error_event}\n\n"
            log.warning("Rate limit hit during stream")

        except LLMError as e:
            error_event = json.dumps({
                "type": "error",
                "content": f"Generation failed: {e.message}",
            })
            yield f"data: {error_event}\n\n"
            log.error(f"LLM error during stream: {e.message}")

        except Exception as e:
            error_event = json.dumps({
                "type": "error",
                "content": "An unexpected error occurred. Please try again.",
            })
            yield f"data: {error_event}\n\n"
            log.exception(f"Unhandled error during stream: {e}")

        finally:
            # Always send done signal so client can stop listening
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
        },
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Raw hybrid search (no generation)",
    description=(
        "Run hybrid search and return raw chunks without LLM generation. "
        "Useful for debugging retrieval quality and tuning chunk strategies."
    ),
)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Raw retrieval endpoint — returns chunks, not an LLM answer.

    This endpoint is extremely useful during development:
      - Check if the right chunks are being retrieved for a query
      - Tune chunk_size and retrieval_top_k settings
      - Debug why an answer is wrong (retrieval issue vs generation issue)
    """
    log.info(f"Search request | query='{request.query[:80]}'")

    metadata_filter = None
    if request.source_filter:
        metadata_filter = {"source": request.source_filter}

    try:
        docs = _search_engine.search(
            query=request.query,
            k=request.k,
            filter=metadata_filter,
        )

        results = [
            {
                "content": doc.page_content[:500],
                "source":  doc.metadata.get("source", "unknown"),
                "page":    doc.metadata.get("page", "?"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "chunk_strategy": doc.metadata.get("chunk_strategy"),
            }
            for doc in docs
        ]

        return SearchResponse(
            results=results,
            total_found=len(results),
            query=request.query,
        )

    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )
    except VectorStoreError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        )