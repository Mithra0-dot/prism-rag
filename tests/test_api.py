"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
tests/test_api.py

Basic API tests for the PRISM FastAPI backend.

These tests verify:
  - Health check endpoint returns 200
  - Document list endpoint works
  - Invalid requests return correct error codes
  - Config loads correctly

Run with:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.config import get_settings

settings = get_settings()

# ── Test client ───────────────────────────────────────────────────────
@pytest.fixture
def client():
    """FastAPI test client — no real server needed."""
    with TestClient(app) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════
# Health Check Tests
# ══════════════════════════════════════════════════════════════════════

def test_health_check_returns_200(client):
    """Health endpoint must always return 200."""
    response = client.get(f"{settings.api_prefix}/health")
    assert response.status_code == 200


def test_health_check_returns_correct_fields(client):
    """Health endpoint must return all expected fields."""
    response = client.get(f"{settings.api_prefix}/health")
    data = response.json()

    assert "status" in data
    assert "app" in data
    assert "version" in data
    assert "environment" in data
    assert data["status"] == "healthy"
    assert data["app"] == "PRISM"


def test_root_endpoint(client):
    """Root endpoint should return app metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "docs" in data


# ══════════════════════════════════════════════════════════════════════
# Document List Tests
# ══════════════════════════════════════════════════════════════════════

def test_list_documents_returns_200(client):
    """Document list endpoint should return 200."""
    response = client.get(f"{settings.api_prefix}/ingest/documents")
    assert response.status_code == 200


def test_list_documents_response_structure(client):
    """Document list must have correct structure."""
    response = client.get(f"{settings.api_prefix}/ingest/documents")
    data = response.json()

    assert "sources" in data
    assert "total_chunks" in data
    assert "collection_name" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["total_chunks"], int)


# ══════════════════════════════════════════════════════════════════════
# Query Validation Tests
# ══════════════════════════════════════════════════════════════════════

def test_query_empty_question_returns_422(client):
    """Empty question should return 422 validation error."""
    response = client.post(
        f"{settings.api_prefix}/query/ask",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_query_missing_question_returns_422(client):
    """Missing question field should return 422."""
    response = client.post(
        f"{settings.api_prefix}/query/ask",
        json={},
    )
    assert response.status_code == 422


def test_search_empty_query_returns_422(client):
    """Empty search query should return 422."""
    response = client.post(
        f"{settings.api_prefix}/query/search",
        json={"query": ""},
    )
    assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════
# Config Tests
# ══════════════════════════════════════════════════════════════════════

def test_settings_load_correctly():
    """Settings must load with correct defaults."""
    assert settings.app_name == "PRISM"
    assert settings.api_prefix == "/api/v1"
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.retrieval_top_k > 0
    assert 0.0 <= settings.hybrid_search_alpha <= 1.0


def test_settings_directories_exist():
    """Required data directories must be created on startup."""
    assert settings.upload_dir.exists()
    assert settings.chroma_persist_dir.exists()


# ══════════════════════════════════════════════════════════════════════
# Upload Validation Tests
# ══════════════════════════════════════════════════════════════════════

def test_upload_no_file_returns_422(client):
    """Upload without a file should return 422."""
    response = client.post(f"{settings.api_prefix}/ingest/upload")
    assert response.status_code == 422


def test_delete_nonexistent_document_returns_404(client):
    """Deleting a document that doesn't exist should return 404."""
    response = client.delete(
        f"{settings.api_prefix}/ingest/document/nonexistent-id-12345"
    )
    assert response.status_code == 404