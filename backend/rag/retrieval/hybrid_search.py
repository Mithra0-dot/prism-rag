"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/rag/retrieval/hybrid_search.py

Hybrid search engine combining BM25 + vector search via RRF fusion.

Why hybrid search beats either method alone:
  ┌─────────────────┬──────────────────────┬────────────────────────┐
  │                 │ BM25 (keyword)       │ Vector (semantic)      │
  ├─────────────────┼──────────────────────┼────────────────────────┤
  │ Strength        │ Exact term matching  │ Meaning/intent capture │
  │                 │ Rare words, IDs,     │ Synonyms, paraphrases  │
  │                 │ product codes        │ Cross-lingual          │
  ├─────────────────┼──────────────────────┼────────────────────────┤
  │ Weakness        │ Misses synonyms      │ Misses exact terms     │
  │                 │ "car" ≠ "automobile" │ Rare words diluted     │
  └─────────────────┴──────────────────────┴────────────────────────┘

  Hybrid search gets the best of both worlds.

Fusion algorithm — Reciprocal Rank Fusion (RRF):
  RRF score = Σ 1 / (k + rank_i)
  where k=60 (smoothing constant) and rank_i is each system's rank.

  RRF is preferred over simple score averaging because:
    1. BM25 and vector scores are on different scales (not comparable)
    2. RRF only uses rank position — scale-invariant by design
    3. Proven in TREC benchmarks to outperform weighted score fusion

alpha parameter (from config, default 0.5):
  Controls the balance. Not used in RRF directly but controls
  how many results each system contributes before fusion:
    alpha=0.0 → only BM25 results
    alpha=0.5 → equal contribution from both (recommended)
    alpha=1.0 → only vector results

Usage:
    from backend.rag.retrieval.hybrid_search import HybridSearchEngine

    engine = HybridSearchEngine(documents=all_chunks)
    results = engine.search("What was the Q3 revenue?", k=5)
"""

from typing import Optional

from langchain.schema import Document

from backend.core.config import get_settings
from backend.core.exceptions import RetrievalError
from backend.core.logger import get_logger
from backend.rag.retrieval.vector_store import get_vector_store

settings = get_settings()
log = get_logger("hybrid_search")


class HybridSearchEngine:
    """
    Hybrid BM25 + Vector search engine with Reciprocal Rank Fusion.

    This class maintains a BM25 index over all ingested document chunks
    and combines its results with ChromaDB vector search using RRF.

    The BM25 index lives in memory and is rebuilt when documents are
    added. For large document sets, this is still fast — BM25 is O(n)
    at query time and the index build is O(n*avg_doc_len).

    Args:
        documents : Initial list of Document chunks to index
                    (can be empty — use add_documents() to add later)
        alpha     : Balance between BM25 and vector (from config)
        rrf_k     : RRF smoothing constant (60 is the standard value)
    """

    # RRF smoothing constant — 60 is the value from the original paper
    # (Cormack, Clarke & Buettcher, SIGIR 2009)
    _RRF_K: int = 60

    def __init__(
        self,
        documents: Optional[list[Document]] = None,
        alpha: Optional[float] = None,
        rrf_k: int = 60,
    ) -> None:
        self.alpha = alpha if alpha is not None else settings.hybrid_search_alpha
        self._rrf_k = rrf_k
        self._vector_store = get_vector_store()

        # BM25 index state
        self._documents: list[Document] = []
        self._bm25_index = None   # lazy-built when documents are added

        if documents:
            self.add_documents(documents)

        log.info(
            f"HybridSearchEngine initialised | "
            f"alpha={self.alpha} | "
            f"rrf_k={self._rrf_k}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the BM25 index.

        The vector store is updated separately via VectorStore.add_documents().
        This method only updates the in-memory BM25 index.

        Args:
            documents : Chunked Document objects from DocumentChunker
        """
        if not documents:
            return

        self._documents.extend(documents)
        self._build_bm25_index()
        log.info(
            f"BM25 index updated | "
            f"total_docs={len(self._documents)}"
        )

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Execute hybrid search and return fused results.

        Pipeline:
          1. Run BM25 keyword search → ranked list A
          2. Run vector similarity search → ranked list B
          3. Apply RRF fusion → unified ranked list
          4. Return top-k results

        Args:
            query  : User's question
            k      : Number of results to return
            filter : Optional metadata filter (passed to vector search)

        Returns:
            List of Documents ranked by RRF fusion score

        Raises:
            RetrievalError : if either search method fails
        """
        k = k or settings.retrieval_top_k

        if not query or not query.strip():
            raise RetrievalError(
                message="Search query cannot be empty",
                detail="empty query string received",
            )

        log.info(f"Hybrid search | query='{query[:60]}' | k={k}")

        # ── Step 1: BM25 search ───────────────────────────────────────
        bm25_results = self._bm25_search(query, k=k * 2)

        # ── Step 2: Vector search ─────────────────────────────────────
        vector_results = self._vector_search(query, k=k * 2, filter=filter)

        # ── Step 3: Fuse with RRF ─────────────────────────────────────
        fused = self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            k=k,
        )

        log.info(
            f"Hybrid search complete | "
            f"bm25={len(bm25_results)} | "
            f"vector={len(vector_results)} | "
            f"fused={len(fused)}"
        )

        return fused

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Like search() but also returns RRF scores for each result.

        Used by the Ragas evaluation pipeline to log retrieval quality
        and by MLflow to track score distributions across experiments.

        Returns:
            List of (Document, rrf_score) tuples ordered by score descending
        """
        k = k or settings.retrieval_top_k
        bm25_results = self._bm25_search(query, k=k * 2)
        vector_results = self._vector_search(query, k=k * 2)

        return self._reciprocal_rank_fusion_with_scores(
            bm25_results=bm25_results,
            vector_results=vector_results,
            k=k,
        )

    # ── BM25 ──────────────────────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        """
        Build BM25 index from all stored documents.

        BM25 (Best Match 25) is a probabilistic ranking function that
        scores documents based on term frequency (TF) and inverse document
        frequency (IDF), with length normalisation.

        Formula: score(D,Q) = Σ IDF(qi) * (tf(qi,D) * (k1+1)) /
                              (tf(qi,D) + k1*(1-b+b*|D|/avgdl))

        where k1=1.5 (term saturation), b=0.75 (length normalisation)

        The rank_bm25 library handles all of this — we just need to
        tokenise our documents (simple whitespace split here).
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RetrievalError(
                message="rank-bm25 not installed. Run: pip install rank-bm25",
                detail=str(e),
            )

        # Tokenise — lowercase + whitespace split
        # In production you'd add stemming/lemmatisation here
        tokenised_corpus = [
            doc.page_content.lower().split()
            for doc in self._documents
        ]

        self._bm25_index = BM25Okapi(tokenised_corpus)
        log.debug(f"BM25 index built | corpus_size={len(self._documents)}")

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        """
        Run BM25 keyword search over the in-memory index.

        Returns top-k documents ranked by BM25 score.
        Falls back gracefully to empty list if index is not built.
        """
        if self._bm25_index is None or not self._documents:
            log.debug("BM25 index empty — skipping keyword search")
            return []

        tokenised_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenised_query)

        # Pair each document with its score and sort descending
        scored_docs = sorted(
            zip(self._documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter out zero-score docs (no keyword overlap at all)
        top_docs = [
            doc for doc, score in scored_docs[:k]
            if score > 0
        ]

        log.debug(f"BM25 returned {len(top_docs)} results for '{query[:40]}'")
        return top_docs

    # ── Vector Search ─────────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        k: int,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Run semantic vector search via ChromaDB.

        Returns top-k documents ranked by cosine similarity
        to the query embedding.
        """
        try:
            return self._vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )
        except Exception as e:
            raise RetrievalError(
                message="Vector search failed during hybrid search",
                detail=str(e),
            ) from e

    # ── RRF Fusion ────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[Document],
        vector_results: list[Document],
        k: int,
    ) -> list[Document]:
        """
        Fuse BM25 and vector results using Reciprocal Rank Fusion.

        RRF Score for a document d:
            RRF(d) = Σ_r 1 / (k + rank_r(d))

        where rank_r(d) is the position of document d in result list r
        (1-indexed), and k=60 is the smoothing constant.

        The smoothing constant k=60 prevents very high-ranked documents
        from dominating. A document ranked #1 scores 1/61 ≈ 0.016,
        while #100 scores 1/160 ≈ 0.006 — a meaningful but not
        overwhelming difference.

        Key insight: we identify documents by their content hash
        (not object identity) so the same chunk appearing in both
        BM25 and vector results gets a combined score boost.
        """
        scores_with_docs: dict[str, tuple[float, Document]] = {}

        # Score BM25 results
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = self._doc_id(doc)
            rrf_score = 1.0 / (self._rrf_k + rank)
            if doc_id in scores_with_docs:
                scores_with_docs[doc_id] = (
                    scores_with_docs[doc_id][0] + rrf_score,
                    doc,
                )
            else:
                scores_with_docs[doc_id] = (rrf_score, doc)

        # Score vector results (additive — same doc gets higher score)
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._doc_id(doc)
            rrf_score = 1.0 / (self._rrf_k + rank)
            if doc_id in scores_with_docs:
                scores_with_docs[doc_id] = (
                    scores_with_docs[doc_id][0] + rrf_score,
                    doc,
                )
            else:
                scores_with_docs[doc_id] = (rrf_score, doc)

        # Sort by RRF score descending and return top-k docs
        sorted_results = sorted(
            scores_with_docs.values(),
            key=lambda x: x[0],
            reverse=True,
        )

        return [doc for _, doc in sorted_results[:k]]

    def _reciprocal_rank_fusion_with_scores(
        self,
        bm25_results: list[Document],
        vector_results: list[Document],
        k: int,
    ) -> list[tuple[Document, float]]:
        """RRF fusion that also returns the scores — used by evaluation."""
        scores_with_docs: dict[str, tuple[float, Document]] = {}

        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = self._doc_id(doc)
            rrf_score = 1.0 / (self._rrf_k + rank)
            if doc_id in scores_with_docs:
                scores_with_docs[doc_id] = (
                    scores_with_docs[doc_id][0] + rrf_score, doc
                )
            else:
                scores_with_docs[doc_id] = (rrf_score, doc)

        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._doc_id(doc)
            rrf_score = 1.0 / (self._rrf_k + rank)
            if doc_id in scores_with_docs:
                scores_with_docs[doc_id] = (
                    scores_with_docs[doc_id][0] + rrf_score, doc
                )
            else:
                scores_with_docs[doc_id] = (rrf_score, doc)

        sorted_results = sorted(
            scores_with_docs.values(),
            key=lambda x: x[0],
            reverse=True,
        )

        return [(doc, score) for score, doc in sorted_results[:k]]

    # ── Helpers ───────────────────────────────────────────────────────

    def _doc_id(self, doc: Document) -> str:
        """
        Generate a stable ID for a document chunk.

        Uses chunk_index from metadata if available (most reliable),
        falls back to a hash of the content. This allows RRF to
        recognise the same chunk appearing in both result lists.
        """
        chunk_index = doc.metadata.get("chunk_index")
        if chunk_index is not None:
            source = doc.metadata.get("source", "unknown")
            return f"{source}::{chunk_index}"

        # Fallback: hash first 200 chars of content
        return str(hash(doc.page_content[:200]))

    @property
    def index_size(self) -> int:
        """Number of documents in the BM25 index."""
        return len(self._documents)

    @property
    def is_ready(self) -> bool:
        """True if the engine has documents indexed and is ready to search."""
        return self._bm25_index is not None and len(self._documents) > 0
