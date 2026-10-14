try:
    import torch
except Exception:
    pass

from fastapi.testclient import TestClient
from src.api.app import app
from src.config import ALL_COLLECTION_NAMES, STRATEGIES


client = TestClient(app)


def test_config_constants():
    assert len(ALL_COLLECTION_NAMES) == 4
    assert "arxiv_semantic_512" in ALL_COLLECTION_NAMES
    assert "hybrid_rerank" in STRATEGIES


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "loaded_collections" in data


def test_collections_endpoint():
    response = client.get("/collections")
    assert response.status_code == 200
    data = response.json()
    assert "available_collections" in data
    assert len(data["available_collections"]) == 4


def test_query_validation_error():
    # Query shorter than min_length=3 should return 422 Unprocessable Entity
    response = client.post(
        "/query",
        json={
            "query": "hi",
            "collection": "arxiv_semantic_512",
            "strategy": "dense",
            "top_k": 3,
            "generate_answer": False,
        },
    )
    assert response.status_code == 422
