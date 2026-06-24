"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/rag/ingestion/chunker.py

Document chunking strategies for the RAG pipeline.

Why chunking matters:
  LLMs and embedding models have token limits. A 50-page PDF cannot
  be embedded as one unit. We split it into overlapping chunks so:
    1. Each chunk fits within the embedding model's context window
    2. Overlapping ensures ideas that span chunk boundaries aren't lost
    3. Smaller chunks = more precise retrieval (less noise per chunk)

Three strategies implemented:
  - RECURSIVE  : LangChain's RecursiveCharacterTextSplitter (default)
                 Tries to split on paragraphs → sentences → words.
                 Best general-purpose strategy.

  - TOKEN      : Splits on exact token counts via tiktoken.
                 Most precise for OpenAI models — respects actual
                 token limits rather than character approximations.

  - SEMANTIC   : Splits based on embedding similarity between sentences.
                 Groups sentences with similar meaning together.
                 Slowest but highest quality — best for dense technical docs.

Usage:
    from backend.rag.ingestion.chunker import DocumentChunker, ChunkStrategy

    chunker = DocumentChunker(strategy=ChunkStrategy.RECURSIVE)
    chunks = chunker.chunk(documents)
"""

from enum import Enum
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import DocumentIngestionError

settings = get_settings()
log = get_logger("chunker")


# ══════════════════════════════════════════════════════════════════════
# Strategy Enum
# ══════════════════════════════════════════════════════════════════════

class ChunkStrategy(str, Enum):
    """
    Available chunking strategies.

    Using str + Enum means these values work directly as FastAPI
    query parameters and in JSON responses without extra serialisation.

    Example API usage:
        POST /api/v1/ingest?strategy=recursive
        POST /api/v1/ingest?strategy=semantic
    """
    RECURSIVE = "recursive"
    TOKEN     = "token"
    SEMANTIC  = "semantic"


# ══════════════════════════════════════════════════════════════════════
# Chunker
# ══════════════════════════════════════════════════════════════════════

class DocumentChunker:
    """
    Splits loaded Documents into chunks ready for embedding.

    The chunker preserves and enriches metadata from the loader:
      - Original source, page, file_type are kept
      - chunk_index is reassigned across the full document set
      - chunk_strategy is recorded (useful for MLflow experiment tracking)

    Args:
        strategy   : ChunkStrategy to use (default: RECURSIVE)
        chunk_size : Target chunk size in tokens/chars (default: from config)
        chunk_overlap : Overlap between chunks (default: from config)
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        log.info(
            f"DocumentChunker initialised | "
            f"strategy={strategy.value} | "
            f"chunk_size={self.chunk_size} | "
            f"overlap={self.chunk_overlap}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of Documents into smaller chunks.

        Args:
            documents : Output from DocumentLoader.load()

        Returns:
            Flat list of chunk Documents with enriched metadata.

        Raises:
            DocumentIngestionError : if chunking fails for any reason
        """
        if not documents:
            log.warning("DocumentChunker received empty document list")
            return []

        log.info(
            f"Chunking {len(documents)} documents "
            f"using '{self.strategy.value}' strategy"
        )

        try:
            if self.strategy == ChunkStrategy.RECURSIVE:
                chunks = self._recursive_chunk(documents)
            elif self.strategy == ChunkStrategy.TOKEN:
                chunks = self._token_chunk(documents)
            elif self.strategy == ChunkStrategy.SEMANTIC:
                chunks = self._semantic_chunk(documents)
            else:
                raise DocumentIngestionError(
                    message=f"Unknown chunking strategy: {self.strategy}"
                )
        except DocumentIngestionError:
            raise
        except Exception as e:
            raise DocumentIngestionError(
                message="Chunking pipeline failed",
                detail=str(e),
            ) from e

        # Re-index chunks across the full document set
        chunks = self._reindex(chunks)

        log.info(
            f"Chunking complete: {len(documents)} pages → "
            f"{len(chunks)} chunks | "
            f"avg_length={self._avg_length(chunks):.0f} chars"
        )

        return chunks

    # ── Strategy Implementations ──────────────────────────────────────

    def _recursive_chunk(self, documents: list[Document]) -> list[Document]:
        """
        RecursiveCharacterTextSplitter — the recommended default.

        Splits in priority order:
          1. Double newlines (paragraph breaks)
          2. Single newlines
          3. Sentence-ending periods
          4. Spaces
          5. Individual characters (last resort)

        This hierarchy means the splitter tries to keep paragraphs
        together, then sentences, only breaking mid-sentence if forced.
        The result is semantically coherent chunks in most cases.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,   # adds 'start_index' to metadata
        )

        chunks = splitter.split_documents(documents)
        self._tag_strategy(chunks)
        return chunks

    def _token_chunk(self, documents: list[Document]) -> list[Document]:
        """
        TokenTextSplitter — splits on exact token boundaries.

        Uses tiktoken (OpenAI's tokenizer) to count tokens precisely.
        This is the most accurate strategy when using OpenAI models
        because chunk_size maps directly to the model's token budget.

        For example, with chunk_size=512 and gpt-4o-mini (128k context),
        retrieval_top_k=5 means we use at most 2,560 tokens for context,
        leaving plenty of room for the system prompt and response.
        """
        try:
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base",  # encoding used by gpt-4o-mini
            )
        except Exception as e:
            raise DocumentIngestionError(
                message="TokenTextSplitter failed — is tiktoken installed?",
                detail=str(e),
            )

        chunks = splitter.split_documents(documents)
        self._tag_strategy(chunks)
        return chunks

    def _semantic_chunk(self, documents: list[Document]) -> list[Document]:
        """
        Semantic chunking — groups sentences by embedding similarity.

        Algorithm:
          1. Split each document into individual sentences
          2. Embed each sentence using the configured embedding model
          3. Compute cosine similarity between consecutive sentences
          4. When similarity drops below a threshold, start a new chunk
             (the drop indicates a topic change)
          5. Merge sentences within each chunk

        This produces chunks that respect topic boundaries rather than
        arbitrary character/token counts. Best for academic papers,
        legal documents, and technical reports with clear section changes.

        Trade-off: slowest strategy — requires embedding every sentence
        twice (once for chunking, once for indexing). Use when quality
        matters more than ingestion speed.
        """

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise DocumentIngestionError(
                message="sentence-transformers required for semantic chunking",
                detail=str(e),
            )

        log.info("Loading embedding model for semantic chunking...")
        model = SentenceTransformer(settings.embedding_model)

        all_chunks: list[Document] = []

        for doc in documents:
            # Split into sentences using regex
            sentences = self._split_sentences(doc.page_content)
            if len(sentences) <= 1:
                all_chunks.append(doc)
                continue

            # Embed all sentences at once (batched for efficiency)
            embeddings = model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,   # cosine sim = dot product
            )

            # Find split points where topic changes (similarity drops)
            split_indices = self._find_semantic_splits(
                embeddings=embeddings,
                threshold=0.3,   # lower = more splits, higher = fewer splits
            )

            # Build chunks from split points
            doc_chunks = self._build_chunks_from_splits(
                sentences=sentences,
                split_indices=split_indices,
                source_doc=doc,
            )
            all_chunks.extend(doc_chunks)

        self._tag_strategy(all_chunks)
        return all_chunks

    # ── Semantic Chunking Helpers ─────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        import re
        # Split on period/exclamation/question mark followed by space + capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_semantic_splits(
        self,
        embeddings,
        threshold: float = 0.3,
    ) -> list[int]:
        """
        Find indices where a new chunk should start.

        Computes cosine similarity between consecutive sentence embeddings.
        A low similarity score indicates a topic shift → split here.
        """
        import numpy as np

        split_indices = [0]  # always start a chunk at the first sentence
        current_chunk_size = 0

        for i in range(1, len(embeddings)):
            # Cosine similarity between consecutive sentences
            similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
            current_chunk_size += len(str(embeddings[i]))

            should_split = (
                similarity < (1 - threshold)           # topic shift detected
                or current_chunk_size > self.chunk_size # hard size limit
            )

            if should_split:
                split_indices.append(i)
                current_chunk_size = 0

        return split_indices

    def _build_chunks_from_splits(
        self,
        sentences: list[str],
        split_indices: list[int],
        source_doc: Document,
    ) -> list[Document]:
        """Merge sentences between split points into Document chunks."""
        chunks: list[Document] = []
        split_indices_set = set(split_indices)

        current_sentences: list[str] = []
        chunk_num = 0

        for i, sentence in enumerate(sentences):
            if i in split_indices_set and current_sentences:
                # Save current chunk
                chunks.append(Document(
                    page_content=" ".join(current_sentences),
                    metadata={
                        **source_doc.metadata,
                        "semantic_chunk_num": chunk_num,
                    },
                ))
                current_sentences = []
                chunk_num += 1

            current_sentences.append(sentence)

        # Flush the final chunk
        if current_sentences:
            chunks.append(Document(
                page_content=" ".join(current_sentences),
                metadata={
                    **source_doc.metadata,
                    "semantic_chunk_num": chunk_num,
                },
            ))

        return chunks

    # ── Post-processing Helpers ───────────────────────────────────────

    def _tag_strategy(self, chunks: list[Document]) -> None:
        """Tag every chunk with the strategy used — recorded in ChromaDB metadata."""
        for chunk in chunks:
            chunk.metadata["chunk_strategy"] = self.strategy.value

    def _reindex(self, chunks: list[Document]) -> list[Document]:
        """Reassign chunk_index sequentially across the full chunk set."""
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        return chunks

    def _avg_length(self, chunks: list[Document]) -> float:
        """Calculate average chunk character length for logging."""
        if not chunks:
            return 0.0
        return sum(len(c.page_content) for c in chunks) / len(chunks)
