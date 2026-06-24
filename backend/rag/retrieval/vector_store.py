"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/rag/retrieval/vector_store.py

ChromaDB vector store interface for PRISM.

Responsibilities:
  - Persist document chunks as embedding vectors in ChromaDB
  - Retrieve the top-k most semantically similar chunks for a query
  - Manage collections (create, list, delete)
  - Expose document metadata for source citation in responses

Why ChromaDB:
  - Runs fully locally — no cloud dependency, no API cost
  - Persists to disk automatically (survives server restarts)
  - Native LangChain integration
  - Fast enough for thousands of documents on a laptop

Architecture note:
  This class wraps LangChain's Chroma wrapper (which itself wraps
  chromadb). The extra layer gives us clean error handling, logging,
  and metadata management without touching ChromaDB's raw API.

Usage:
    from backend.rag.retrieval.vector_store import VectorStore

    store = VectorStore()

    # Store chunks after ingestion
    store.add_documents(chunks)

    # Retrieve relevant chunks for a query
    results = store.similarity_search("What was the Q3 revenue?", k=5)
"""

import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from backend.core.config import get_settings
from backend.core.exceptions import DocumentNotFoundError, VectorStoreError
from backend.core.logger import get_logger
from backend.rag.retrieval.embedder import get_embedder

settings = get_settings()
log = get_logger("vector_store")


class VectorStore:
    """
    ChromaDB-backed vector store for PRISM document chunks.

    Provides a clean interface for:
      - Adding document chunks (with deduplication)
      - Semantic similarity search
      - Collection management
      - Document metadata queries

    The store is persistent — data survives server restarts because
    ChromaDB writes to disk at settings.chroma_persist_dir.

    Args:
        collection_name : ChromaDB collection to use (default: from config)
        persist_dir     : Directory for ChromaDB persistence (default: from config)
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self._embedder = get_embedder()
        self._store: Optional[Chroma] = None   # lazy-loaded

        log.info(
            f"VectorStore configured | "
            f"collection={self.collection_name} | "
            f"persist_dir={self.persist_dir}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: list[Document],
        source_id: Optional[str] = None,
    ) -> list[str]:
        """
        Add document chunks to the vector store.

        Embeds each chunk and stores it with its metadata.
        Returns the list of generated document IDs.

        Args:
            documents : Chunked Document objects from DocumentChunker
            source_id : Optional identifier to group chunks by document
                        (used for targeted deletion later)

        Returns:
            List of ChromaDB document IDs

        Raises:
            VectorStoreError : if storage operation fails
        """
        if not documents:
            log.warning("add_documents called with empty list")
            return []

        # Tag all chunks with source_id for grouped deletion
        if source_id:
            for doc in documents:
                doc.metadata["source_id"] = source_id

        log.info(
            f"Adding {len(documents)} chunks to "
            f"collection '{self.collection_name}'..."
        )
        start = time.perf_counter()

        try:
            store = self._get_store()
            ids = store.add_documents(documents)
        except Exception as e:
            raise VectorStoreError(
                message="Failed to add documents to ChromaDB",
                detail=str(e),
            ) from e

        elapsed = time.perf_counter() - start
        log.info(
            f"Stored {len(ids)} chunks in {elapsed:.2f}s | "
            f"collection='{self.collection_name}'"
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Find the k most semantically similar chunks to a query.

        This is pure vector search — used by the hybrid search module
        which combines this with BM25 keyword search.

        Args:
            query  : The user's question or search text
            k      : Number of results to return (default: from config)
            filter : Optional ChromaDB metadata filter dict
                     e.g. {"source": "annual_report.pdf"} to search
                     only within a specific document

        Returns:
            List of Document objects ordered by similarity (most similar first)

        Raises:
            VectorStoreError : if the search operation fails
        """
        k = k or settings.retrieval_top_k

        log.debug(f"Similarity search | query='{query[:60]}...' | k={k}")
        start = time.perf_counter()

        try:
            store = self._get_store()
            results = store.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )
        except Exception as e:
            raise VectorStoreError(
                message="Similarity search failed",
                detail=str(e),
            ) from e

        elapsed = time.perf_counter() - start
        log.debug(
            f"Retrieved {len(results)} chunks in {elapsed*1000:.1f}ms"
        )

        return results

    def similarity_search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """
        Like similarity_search but also returns relevance scores.

        Scores are cosine similarity values between 0.0 and 1.0.
        Higher = more similar. Used by the hybrid search module
        to combine vector scores with BM25 scores.

        Returns:
            List of (Document, score) tuples, ordered by score descending
        """
        k = k or settings.retrieval_top_k

        try:
            store = self._get_store()
            results = store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter,
            )
        except Exception as e:
            raise VectorStoreError(
                message="Similarity search with scores failed",
                detail=str(e),
            ) from e

        return results

    def delete_document(self, source_id: str) -> None:
        """
        Delete all chunks belonging to a specific document.

        Uses the source_id metadata field set during add_documents.
        Called when a user re-uploads a document to prevent duplicates.

        Args:
            source_id : The document identifier used during ingestion

        Raises:
            DocumentNotFoundError : if no chunks found with this source_id
            VectorStoreError      : if deletion fails
        """
        log.info(f"Deleting document chunks | source_id='{source_id}'")

        try:
            store = self._get_store()
            collection = store._collection

            # Find existing chunks with this source_id
            existing = collection.get(
                where={"source_id": source_id},
                include=["metadatas"],
            )

            if not existing["ids"]:
                raise DocumentNotFoundError(
                    message=f"No document found with source_id='{source_id}'",
                    detail="Nothing to delete",
                )

            collection.delete(where={"source_id": source_id})
            log.info(
                f"Deleted {len(existing['ids'])} chunks "
                f"for source_id='{source_id}'"
            )

        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise VectorStoreError(
                message=f"Failed to delete document '{source_id}'",
                detail=str(e),
            ) from e

    def get_collection_stats(self) -> dict:
        """
        Return statistics about the current ChromaDB collection.

        Used by:
          - The health check endpoint
          - The Streamlit sidebar (shows document count)
          - MLflow experiment logging

        Returns:
            Dict with chunk_count, collection_name, persist_dir
        """
        try:
            store = self._get_store()
            count = store._collection.count()
            return {
                "collection_name": self.collection_name,
                "chunk_count": count,
                "persist_dir": self.persist_dir,
                "embedding_model": self._embedder.model_name,
                "embedding_dimension": self._embedder.dimension,
            }
        except Exception as e:
            raise VectorStoreError(
                message="Failed to get collection statistics",
                detail=str(e),
            ) from e

    def list_sources(self) -> list[str]:
        """
        Return a list of unique document sources in the collection.

        Used by the Streamlit UI to show which documents have been
        ingested, and to let users filter searches by document.

        Returns:
            Sorted list of unique source filenames
        """
        try:
            store = self._get_store()
            results = store._collection.get(include=["metadatas"])
            sources = {
                meta.get("source", "unknown")
                for meta in results["metadatas"]
                if meta
            }
            return sorted(sources)
        except Exception as e:
            raise VectorStoreError(
                message="Failed to list document sources",
                detail=str(e),
            ) from e

    def collection_exists(self) -> bool:
        """Check if the collection has any documents stored."""
        try:
            stats = self.get_collection_stats()
            return stats["chunk_count"] > 0
        except Exception:
            return False

    # ── Internal ──────────────────────────────────────────────────────

    def _get_store(self) -> Chroma:
        """
        Lazy-load the ChromaDB store.

        ChromaDB initialisation opens a connection to the persistent
        storage directory. We do this lazily so the import of this
        module doesn't immediately open database connections.
        """
        if self._store is None:
            log.info(
                f"Initialising ChromaDB | "
                f"collection='{self.collection_name}' | "
                f"persist_dir='{self.persist_dir}'"
            )
            try:
                self._store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self._embedder,
                    persist_directory=self.persist_dir,
                )
            except Exception as e:
                raise VectorStoreError(
                    message="Failed to initialise ChromaDB",
                    detail=str(e),
                ) from e

            log.info("ChromaDB initialised successfully")

        return self._store


# ══════════════════════════════════════════════════════════════════════
# Singleton factory
# ══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """
    Return a cached singleton VectorStore instance.

    Ensures only one ChromaDB connection exists per server process.
    Import and call this anywhere you need vector store access:

        from backend.rag.retrieval.vector_store import get_vector_store
        store = get_vector_store()
    """
    return VectorStore()
