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
    settings = settings or Settings(
        chroma_db_path=tmp_path / "chroma_db",
        bm25_index_path=tmp_path / "bm25_index.json",
    )
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
    _write_note(export_dir, "20240115", "Note One", "p1", "The answer is 42")
    _write_note(export_dir, "20240116", "Note Two", "p2", "Second note text")

    client = _make_client(tmp_path)

    response = client.post("/ingest", json={"export_dir": str(export_dir)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "delta"
    assert payload["changed_notes"] == 2
    assert payload["removed_notes"] == 0
    assert payload["chunks_indexed"] > 0
    assert payload["delta_summary"]["added_notes"] == 2
    assert payload["delta_summary"]["updated_notes"] == 0

    stats = client.get("/stats")
    assert stats.status_code == 200
    stats_payload = stats.json()
    assert stats_payload["chunks"] > 0
    assert stats_payload["notes_indexed"] == 2
    assert stats_payload["last_sync_time"] is not None
    assert stats_payload["last_ingest"]["mode"] == "delta"
    assert stats_payload["delta_summary"]["added_notes"] == 2
    assert stats_payload["configured_models"]["embed_model"] == "nomic-embed-text"
    assert stats_payload["store_size_bytes"] >= 0


def test_ask_returns_answer(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")

    llm = FakeLLM()
    llm.set_response("The answer is 42 [1]")
    client = _make_client(tmp_path, llm=llm)

    ingest = client.post("/ingest", json={"export_dir": str(export_dir)})
    assert ingest.status_code == 200

    response = client.post("/ask", json={"query": "The answer is 42"})
    assert response.status_code == 200
    payload = response.json()
    assert "42" in payload["answer"]
    assert payload["query"] == "The answer is 42"


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
    assert payload["results"][0]["note_id"] == "p1"
    assert payload["results"][0]["source"] == {"note_id": "p1"}


def test_health_endpoint(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["embedder"], bool)
    assert isinstance(payload["llm"], bool)
    assert isinstance(payload["store"], bool)


def test_ingest_cors_preflight(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.options(
        "/ingest",
        headers={
            "Origin": "tauri://localhost",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"


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


def test_reindex_clears_store(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")

    store = InMemoryStore()
    client = _make_client(tmp_path, store=store)

    ingest = client.post("/ingest", json={"export_dir": str(export_dir)})
    assert ingest.status_code == 200
    assert store.size() > 0

    # Remove files and reindex to clear store
    for html_file in export_dir.glob("*.html"):
        html_file.unlink()

    reindex = client.post("/ingest", json={"export_dir": str(export_dir), "reindex": True})
    assert reindex.status_code == 200
    assert reindex.json()["mode"] == "full"
    assert store.size() == 0


def test_ingest_mode_full_then_delta(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    note_file = export_dir / "20240115 Note One [p1].html"
    note_file.write_text("<html><body>Hello world</body></html>", encoding="utf-8")

    client = _make_client(tmp_path)

    full = client.post("/ingest", json={"export_dir": str(export_dir), "mode": "full"})
    assert full.status_code == 200
    full_payload = full.json()
    assert full_payload["mode"] == "full"
    assert full_payload["delta_summary"]["added_notes"] == 1

    delta = client.post("/ingest", json={"export_dir": str(export_dir), "mode": "delta"})
    assert delta.status_code == 200
    delta_payload = delta.json()
    assert delta_payload["mode"] == "delta"
    assert delta_payload["changed_notes"] == 0
    assert delta_payload["delta_summary"]["unchanged_notes"] == 1
