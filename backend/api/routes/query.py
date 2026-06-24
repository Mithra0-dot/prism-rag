"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/api/routes/query.py

RAG query API endpoints.
"""

import json
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.core.exceptions import (
    LLMError,
    RateLimitError,
    RetrievalError,
    VectorStoreError,
)
from backend.core.logger import get_logger
from backend.rag.generation.chain import PRISMChain
from backend.rag.generation.prompt import PromptType
from backend.rag.retrieval.hybrid_search import HybridSearchEngine
from backend.rag.retrieval.vector_store import get_vector_store

settings = get_settings()
log = get_logger("query_route")

router = APIRouter(prefix="/query", tags=["Query"])

_vector_store = get_vector_store()
_search_engine = HybridSearchEngine()
_chain = PRISMChain(search_engine=_search_engine)


# ══════════════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about your documents",
        examples=["What was the revenue in Q3?"],
    )
    k: Optional[int] = Field(default=None, ge=1, le=20)
    source_filter: Optional[str] = Field(default=None)
    prompt_type: PromptType = Field(default=PromptType.QA)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    source_filter: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int
    latency_seconds: float


class SearchResponse(BaseModel):
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
)
async def ask(request: QueryRequest) -> QueryResponse:
    """Non-streaming RAG Q&A endpoint."""
    log.info(f"Query request | question='{request.question[:80]}'")

    try:
        chain = PRISMChain(
            prompt_type=request.prompt_type,
            search_engine=_search_engine,
        )
        result = chain.query(question=request.question, k=request.k)

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
    response_class=StreamingResponse,
)
async def stream(request: QueryRequest) -> StreamingResponse:
    """Streaming RAG Q&A via Server-Sent Events (SSE)."""
    log.info(f"Stream request | question='{request.question[:80]}'")

    async def event_generator() -> AsyncIterator[str]:
        chain = PRISMChain(
            prompt_type=request.prompt_type,
            search_engine=_search_engine,
        )

        try:
            async for token in chain.astream(
                question=request.question,
                k=request.k,
                filter={"source": request.source_filter} if request.source_filter else None,
            ):
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"

            sources = chain.get_sources()
            sources_event = json.dumps({"type": "sources", "content": sources})
            yield f"data: {sources_event}\n\n"

        except RetrievalError as e:
            error_event = json.dumps({
                "type": "error",
                "content": f"Retrieval failed: {e.message}",
            })
            yield f"data: {error_event}\n\n"

        except RateLimitError as e:
            error_event = json.dumps({"type": "error", "content": e.message})
            yield f"data: {error_event}\n\n"

        except LLMError as e:
            error_event = json.dumps({
                "type": "error",
                "content": f"Generation failed: {e.message}",
            })
            yield f"data: {error_event}\n\n"

        except Exception as e:
            error_event = json.dumps({
                "type": "error",
                "content": "An unexpected error occurred. Please try again.",
            })
            yield f"data: {error_event}\n\n"
            log.exception(f"Unhandled error during stream: {e}")

        finally:
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Raw hybrid search (no generation)",
)
async def search(request: SearchRequest) -> SearchResponse:
    """Raw retrieval endpoint — returns chunks without LLM generation."""
    log.info(f"Search request | query='{request.query[:80]}'")

    try:
        docs = _search_engine.search(
            query=request.query,
            k=request.k,
            filter={"source": request.source_filter} if request.source_filter else None,
        )

        results = [
            {
                "content": doc.page_content[:500],
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
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
