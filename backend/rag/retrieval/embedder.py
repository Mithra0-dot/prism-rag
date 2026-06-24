"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/rag/retrieval/embedder.py

Embedding model wrapper for PRISM.

Responsibilities:
  - Load and cache the sentence-transformers model (loaded once, reused)
  - Implement LangChain's Embeddings interface for ChromaDB compatibility
  - Embed both documents (batched) and queries (single)
  - Expose model metadata for MLflow experiment tracking

Why sentence-transformers:
  - Runs 100% locally — no API key, no cost, no rate limits
  - all-MiniLM-L6-v2 is 80MB, fast, and produces 384-dim vectors
  - Scores 58.8 on MTEB benchmark — strong for a model this small
  - Can be swapped for OpenAI embeddings by changing one config value

Usage:
    from backend.rag.retrieval.embedder import PRISMEmbedder

    embedder = PRISMEmbedder()

    # Embed a single query
    vector = embedder.embed_query("What is the revenue for Q3?")

    # Embed multiple document chunks (batched)
    vectors = embedder.embed_documents(["chunk one...", "chunk two..."])
"""

import time
from functools import lru_cache
from typing import Optional

from langchain.embeddings.base import Embeddings

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import EmbeddingError

settings = get_settings()
log = get_logger("embedder")


class PRISMEmbedder(Embeddings):
    """
    Production embedding wrapper for PRISM.

    Implements LangChain's Embeddings abstract base class so this class
    can be passed directly to ChromaDB, FAISS, or any other LangChain
    vector store without modification.

    The underlying model is loaded lazily on first use and cached —
    loading a sentence-transformer model takes ~2 seconds, so we never
    want to load it more than once per server process.

    Args:
        model_name    : HuggingFace model ID (default: from config)
        batch_size    : Number of texts to embed in one forward pass
        normalize     : L2-normalise vectors (enables cosine via dot product)
        show_progress : Show tqdm progress bar during batch embedding
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        self._model = None   # lazy-loaded on first embed call

        log.info(f"PRISMEmbedder configured | model={self.model_name}")

    # ── LangChain Embeddings Interface ────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document chunks.

        Called by ChromaDB during ingestion to build the vector index.
        Processes texts in batches for memory efficiency.

        Args:
            texts : List of chunk text strings

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            EmbeddingError : if the model fails to encode any input
        """
        if not texts:
            log.warning("embed_documents called with empty list")
            return []

        model = self._get_model()
        log.info(f"Embedding {len(texts)} document chunks...")
        start = time.perf_counter()

        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
            )
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed document chunks",
                detail=str(e),
            ) from e

        elapsed = time.perf_counter() - start
        log.info(
            f"Embedded {len(texts)} chunks in {elapsed:.2f}s "
            f"({elapsed/len(texts)*1000:.1f}ms per chunk)"
        )

        # Convert numpy array to plain Python list for JSON serialisability
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single user query.

        Called during retrieval to convert the user's question into a
        vector that can be compared against document chunk vectors.

        Uses the same model and normalisation as embed_documents so
        that dot product gives cosine similarity directly.

        Args:
            text : The user's query string

        Returns:
            Single embedding vector as a list of floats

        Raises:
            EmbeddingError : if text is empty or model fails
        """
        if not text or not text.strip():
            raise EmbeddingError(
                message="Cannot embed an empty query",
                detail="text was empty or whitespace only",
            )

        model = self._get_model()
        log.debug(f"Embedding query: '{text[:80]}...'")

        try:
            vector = model.encode(
                text.strip(),
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed query",
                detail=str(e),
            ) from e

        return vector.tolist()

    # ── Model Management ──────────────────────────────────────────────

    def _get_model(self):
        """
        Lazy-load the sentence-transformer model.

        The model is loaded on the first embed call and stored as an
        instance variable. Subsequent calls reuse the loaded model.

        This pattern avoids loading the model at import time (which
        would slow down server startup) and avoids reloading it on
        every request (which would be extremely slow).
        """
        if self._model is None:
            log.info(f"Loading embedding model: {self.model_name}")
            start = time.perf_counter()

            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self._get_device(),
                )
            except Exception as e:
                raise EmbeddingError(
                    message=f"Failed to load embedding model '{self.model_name}'",
                    detail=str(e),
                ) from e

            elapsed = time.perf_counter() - start
            log.info(
                f"Embedding model loaded in {elapsed:.2f}s | "
                f"device={self._get_device()} | "
                f"dimension={self.dimension}"
            )

        return self._model

    def _get_device(self) -> str:
        """
        Auto-detect the best available compute device.

        Returns 'cuda' if a GPU is available (much faster for large batches),
        otherwise falls back to 'cpu'. For sentence-transformers with
        all-MiniLM-L6-v2, CPU is fast enough for our use case.
        """
        try:
            import torch
            if torch.cuda.is_available():
                log.info("GPU detected — using CUDA for embeddings")
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # ── Properties ────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """
        Returns the vector dimension of the current embedding model.
        Must match the ChromaDB collection's embedding dimension.
        """
        return settings.embedding_dimension

    @property
    def model_info(self) -> dict:
        """
        Returns model metadata for MLflow experiment tracking.

        Logged at the start of each RAG experiment run so results
        can be compared across different embedding models.
        """
        return {
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "device": self._get_device(),
        }


# ══════════════════════════════════════════════════════════════════════
# Module-level singleton factory
# ══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_embedder() -> PRISMEmbedder:
    """
    Return a cached singleton PRISMEmbedder instance.

    Using @lru_cache ensures the embedding model is loaded exactly once
    per server process regardless of how many requests come in.

    Usage in any module:
        from backend.rag.retrieval.embedder import get_embedder
        embedder = get_embedder()
        vector = embedder.embed_query("my question")
    """
    return PRISMEmbedder()