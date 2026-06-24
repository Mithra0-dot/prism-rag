"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/core/config.py

Central configuration management using Pydantic BaseSettings.
All environment variables are validated and typed here.
Every other module imports from this file — never from os.environ directly.
"""

from pathlib import Path
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root (two levels up from this file) ──────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    PRISM application settings.

    Values are loaded in this priority order:
      1. Environment variables
      2. .env file at project root
      3. Defaults defined here

    Usage:
        from backend.core.config import get_settings
        settings = get_settings()
        print(settings.openai_api_key)
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── App metadata ─────────────────────────────────────────────────
    app_name: str = "PRISM"
    app_version: str = "1.0.0"
    app_description: str = (
        "Precision Retrieval with Intelligent Semantic Multimodal RAG Engine"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(
        default="development",
        description="One of: development | staging | production",
    )

    # ── API keys ─────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for embeddings and LLM generation",
    )
    huggingface_token: str = Field(
        default="",
        description="HuggingFace token for private model access",
    )

    # ── LLM settings ─────────────────────────────────────────────────
    llm_provider: str = Field(
        default="openai",
        description="LLM backend: 'openai' or 'huggingface'",
    )
    llm_model_name: str = Field(
        default="gpt-4o-mini",
        description="Model name for the selected provider",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature — lower = more deterministic",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Maximum tokens in the generated response",
    )

    # ── Embedding settings ────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model (free, runs locally)",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Vector dimension — must match the embedding model",
    )

    # ── ChromaDB settings ─────────────────────────────────────────────
    chroma_persist_dir: Path = Field(
        default=BASE_DIR / "data" / "chroma_db",
        description="Directory where ChromaDB persists its data",
    )
    chroma_collection_name: str = Field(
        default="prism_documents",
        description="ChromaDB collection name",
    )

    # ── Chunking settings ─────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Target token size for each document chunk",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Token overlap between consecutive chunks",
    )

    # ── Retrieval settings ────────────────────────────────────────────
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve per query",
    )
    hybrid_search_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight balance for hybrid search. "
            "0.0 = pure BM25 keyword, 1.0 = pure vector semantic"
        ),
    )

    # ── File upload settings ──────────────────────────────────────────
    upload_dir: Path = Field(
        default=BASE_DIR / "data" / "uploads",
        description="Temporary storage for uploaded files",
    )
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum allowed file upload size in megabytes",
    )
    allowed_extensions: list[str] = Field(
        default=[".pdf", ".txt", ".md", ".docx"],
        description="Permitted file extensions for document ingestion",
    )

    # ── MLflow settings ───────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default=str(BASE_DIR / "data" / "mlflow"),
        description="Local directory or remote URI for MLflow tracking",
    )
    mlflow_experiment_name: str = Field(
        default="PRISM-RAG-Experiments",
        description="MLflow experiment name for grouping runs",
    )

    # ── FastAPI server settings ───────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_prefix: str = Field(
        default="/api/v1",
        description="Global URL prefix for all API routes",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:8501"],   # Streamlit default port
        description="Allowed CORS origins",
    )

    # ── Streamlit settings ────────────────────────────────────────────
    streamlit_host: str = Field(default="localhost")
    streamlit_port: int = Field(default=8501)
    backend_url: str = Field(
        default="http://localhost:8000",
        description="Full URL the Streamlit frontend uses to reach FastAPI",
    )

    # ── Validators ────────────────────────────────────────────────────
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"openai", "huggingface"}
        if v.lower() not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("chroma_persist_dir", "upload_dir", mode="before")
    @classmethod
    def ensure_directory_exists(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Computed helpers ──────────────────────────────────────────────
    @property
    def max_upload_size_bytes(self) -> int:
        """Upload limit in bytes — used by FastAPI's UploadFile validation."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Using @lru_cache means the .env file is read exactly once
    at application startup — not on every function call.

    Usage in any module:
        from backend.core.config import get_settings
        settings = get_settings()
    """
    return Settings()
