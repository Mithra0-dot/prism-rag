"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/rag/generation/chain.py

LangChain RAG chain — updated to use Ollama (free, local LLM).
"""

import time
from typing import AsyncIterator, Optional

from langchain.schema import Document
from langchain_community.chat_models import ChatOllama

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import LLMError, RetrievalError
from backend.rag.retrieval.hybrid_search import HybridSearchEngine
from backend.rag.generation.prompt import (
    get_rag_prompt,
    format_context,
    PromptType,
)

settings = get_settings()
log = get_logger("chain")


class PRISMChain:
    """
    End-to-end RAG chain for PRISM using Ollama (local, free).

    Pipeline:
      User Query → HybridSearch → format_context → Ollama LLM → Stream tokens

    Args:
        prompt_type   : Which prompt template to use
        search_engine : HybridSearchEngine instance
        temperature   : LLM sampling temperature
        enable_memory : Track conversation history across turns
    """

    def __init__(
        self,
        prompt_type: PromptType = PromptType.QA,
        search_engine: Optional[HybridSearchEngine] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_memory: bool = False,
    ) -> None:
        self.prompt_type = prompt_type
        self.enable_memory = enable_memory
        self._chat_history: list[dict] = []
        self._last_retrieved_docs: list[Document] = []

        self._search_engine = search_engine or HybridSearchEngine()
        self._llm = self._build_llm(temperature=temperature or settings.llm_temperature)
        self._prompt = get_rag_prompt(prompt_type)

        log.info(
            f"PRISMChain initialised | "
            f"provider=ollama | "
            f"model=llama3.2 | "
            f"prompt={prompt_type.value}"
        )

    # ── Public API ────────────────────────────────────────────────────

    async def astream(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        """Stream the RAG response token by token."""
        if not question or not question.strip():
            raise LLMError(message="Question cannot be empty")

        log.info(f"Streaming response for: '{question[:80]}'")
        start = time.perf_counter()

        # 1. Retrieve
        try:
            docs = self._search_engine.search(
                query=question,
                k=k or settings.retrieval_top_k,
                filter=filter,
            )
            self._last_retrieved_docs = docs
        except Exception as e:
            raise RetrievalError(
                message="Failed to retrieve context for your question",
                detail=str(e),
            ) from e

        log.info(f"Retrieved {len(docs)} chunks for generation")

        # 2. Format context
        context = format_context(docs)

        # 3. Build prompt
        prompt_input = self._build_prompt_input(question=question, context=context)

        # 4. Stream response
        full_response = ""
        try:
            async for chunk in self._llm.astream(
                self._prompt.format_messages(**prompt_input)
            ):
                token = chunk.content
                full_response += token
                yield token

        except Exception as e:
            raise LLMError(
                message="LLM generation failed. Is Ollama running?",
                detail=str(e),
            ) from e

        elapsed = time.perf_counter() - start
        log.info(f"Response streamed in {elapsed:.2f}s")

        if self.enable_memory:
            self._update_history(question, full_response)

    def query(self, question: str, k: Optional[int] = None) -> dict:
        """Synchronous RAG query — returns full response + sources."""
        import asyncio

        async def _run():
            tokens = []
            async for token in self.astream(question, k=k):
                tokens.append(token)
            return "".join(tokens)

        start = time.perf_counter()
        answer = asyncio.run(_run())
        latency = time.perf_counter() - start

        return {
            "answer": answer,
            "sources": self._extract_sources(self._last_retrieved_docs),
            "chunks": [d.page_content for d in self._last_retrieved_docs],
            "latency": round(latency, 3),
        }

    def get_sources(self) -> list[dict]:
        """Return source citations from the last retrieval."""
        return self._extract_sources(self._last_retrieved_docs)

    def clear_history(self) -> None:
        self._chat_history = []
        log.info("Conversation history cleared")

    # ── LLM Builder ───────────────────────────────────────────────────

    def _build_llm(self, temperature: float) -> ChatOllama:
        """
        Build a ChatOllama instance pointing to the local Ollama server.

        Ollama runs as a local server on port 11434 by default.
        Make sure Ollama is running before starting PRISM:
            ollama serve   (runs automatically on Windows after install)
        """
        return ChatOllama(
            model="llama3.2",
            temperature=temperature,
            base_url="http://localhost:11434",
        )

    # ── Prompt Helpers ────────────────────────────────────────────────

    def _build_prompt_input(self, question: str, context: str) -> dict:
        base = {"context": context, "question": question}
        if self.prompt_type == PromptType.CONVERSATIONAL:
            base["chat_history"] = self._format_history()
        return base

    def _format_history(self) -> str:
        if not self._chat_history:
            return "No previous conversation."
        lines = []
        for turn in self._chat_history[-6:]:
            lines.append(f"Human: {turn['question']}")
            lines.append(f"Assistant: {turn['answer'][:200]}...")
        return "\n".join(lines)

    def _update_history(self, question: str, answer: str) -> None:
        self._chat_history.append({"question": question, "answer": answer})
        if len(self._chat_history) > 10:
            self._chat_history = self._chat_history[-10:]

    # ── Source Extraction ─────────────────────────────────────────────

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        seen = set()
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page   = doc.metadata.get("page", "?")
            key    = f"{source}::{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename":  source,
                    "page":      page,
                    "snippet":   doc.page_content[:150].strip() + "...",
                    "file_type": doc.metadata.get("file_type", "unknown"),
                })
        return sources
