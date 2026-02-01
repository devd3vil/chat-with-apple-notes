"""
Tests for FastAPI endpoints.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api import create_app
from src.chain import Chunker
from src.config import Settings
from src.embedder import FakeEmbedder
from src.llm import FakeLLM
from src.store import InMemoryStore
from src.sync import SyncState


def _write_note(export_dir: Path, date: str, title: str, note_id: str, body: str) -> None:
    filename = f"{date} {title} [{note_id}].html"
    html = f"<html><body>{body}</body></html>"
    (export_dir / filename).write_text(html, encoding="utf-8")


def _make_client(
    tmp_path: Path,
    llm: FakeLLM | None = None,
    settings: Settings | None = None,
    store: InMemoryStore | None = None,
) -> TestClient:
    settings = settings or Settings(chroma_db_path=tmp_path / "chroma_db")
    embedder = FakeEmbedder(dimension=3)
    store = store or InMemoryStore()
    llm = llm or FakeLLM()
    chunker = Chunker(chunk_size=20, chunk_overlap=5)
    sync_state = SyncState(state_file=tmp_path / "sync_state.json")

    app = create_app(
        settings=settings,
        embedder=embedder,
        store=store,
        llm=llm,
        chunker=chunker,
        sync_state=sync_state,
    )
    return TestClient(app)


def test_ingest_and_stats(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")
    _write_note(export_dir, "20240116", "Note Two", "p2", "Second note text")

    client = _make_client(tmp_path)

    response = client.post("/ingest", json={"export_dir": str(export_dir)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["changed_notes"] == 2
    assert payload["removed_notes"] == 0
    assert payload["chunks_indexed"] > 0

    stats = client.get("/stats")
    assert stats.status_code == 200
    stats_payload = stats.json()
    assert stats_payload["chunks"] > 0
    assert stats_payload["last_sync_time"] is not None


def test_ask_returns_answer(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")

    llm = FakeLLM()
    llm.set_response("The answer is 42 [1]")
    client = _make_client(tmp_path, llm=llm)

    ingest = client.post("/ingest", json={"export_dir": str(export_dir)})
    assert ingest.status_code == 200

    response = client.post("/ask", json={"query": "What is the answer?"})
    assert response.status_code == 200
    payload = response.json()
    assert "42" in payload["answer"]
    assert payload["query"] == "What is the answer?"


def test_search_returns_results(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")

    client = _make_client(tmp_path)
    ingest = client.post("/ingest", json={"export_dir": str(export_dir)})
    assert ingest.status_code == 200

    response = client.post("/search", json={"query": "Hello world", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "Hello world"
    assert payload["top_k"] == 3
    assert len(payload["results"]) >= 1
    assert payload["results"][0]["chunk_id"].startswith("p1_")


def test_health_endpoint(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["embedder"], bool)
    assert isinstance(payload["llm"], bool)
    assert isinstance(payload["store"], bool)


def test_auto_ingest_on_startup(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")

    settings = Settings(
        chroma_db_path=tmp_path / "chroma_db",
        notes_export_dir=export_dir,
        auto_ingest_on_startup=True,
    )
    store = InMemoryStore()
    client = _make_client(tmp_path, settings=settings, store=store)

    with client:
        assert store.size() > 0
