"""
Tests for FastAPI endpoints.
"""

import time
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
    embedder: FakeEmbedder | None = None,
) -> TestClient:
    settings = settings or Settings(
        chroma_db_path=tmp_path / "chroma_db",
        bm25_index_path=tmp_path / "bm25_index.json",
        ollama_embedding_model="nomic-embed-text",
        ollama_chat_model="neural-chat",
        rerank_enabled=False,
        rerank_warmup_enabled=False,
    )
    embedder = embedder or FakeEmbedder(dimension=3)
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


def _wait_for_job_terminal(client: TestClient, job_id: str, timeout_s: float = 6.0) -> dict:
    deadline = time.time() + timeout_s
    last_payload: dict = {}
    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] in {"completed", "failed", "cancelled"}:
            return last_payload
        time.sleep(0.03)
    return last_payload


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
        ollama_embedding_model="nomic-embed-text",
        ollama_chat_model="neural-chat",
        rerank_enabled=False,
        rerank_warmup_enabled=False,
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


def test_jobs_ingest_completes_and_reports_progress(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")
    _write_note(export_dir, "20240116", "Note Two", "p2", "More text")

    client = _make_client(tmp_path)
    started = client.post("/jobs/ingest", json={"export_dir": str(export_dir), "mode": "full"})
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    terminal = _wait_for_job_terminal(client, job_id)
    assert terminal["status"] == "completed"
    assert terminal["phase"] == "finalizing"
    assert terminal["result"]["mode"] == "full"
    assert terminal["result"]["changed_notes"] >= 1


def test_jobs_ingest_cancelled(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    large = "hello world " * 600
    for idx in range(1, 26):
        _write_note(export_dir, f"202401{idx:02d}", f"Note {idx}", f"p{idx}", large)

    class SlowFakeEmbedder(FakeEmbedder):
        def embed(self, text: str) -> list[float]:
            time.sleep(0.005)
            return super().embed(text)

    client = _make_client(tmp_path, embedder=SlowFakeEmbedder(dimension=3))

    started = client.post("/jobs/ingest", json={"export_dir": str(export_dir), "mode": "full"})
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    cancelled = client.post(f"/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200

    terminal = _wait_for_job_terminal(client, job_id, timeout_s=8.0)
    assert terminal["status"] == "cancelled"


def test_reset_notes_index_clears_store_and_manifest(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(export_dir, "20240115", "Note One", "p1", "Hello world")

    client = _make_client(tmp_path)
    ingest = client.post("/ingest", json={"export_dir": str(export_dir), "mode": "full"})
    assert ingest.status_code == 200

    stats_before = client.get("/stats")
    assert stats_before.status_code == 200
    assert stats_before.json()["notes_indexed"] >= 1

    reset = client.post("/reset_notes_index")
    assert reset.status_code == 200
    assert reset.json()["status"] == "ok"

    stats_after = client.get("/stats")
    assert stats_after.status_code == 200
    assert stats_after.json()["chunks"] == 0
    assert stats_after.json()["notes_indexed"] == 0


def test_jobs_invalid_id_returns_404(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    get_resp = client.get("/jobs/does-not-exist")
    assert get_resp.status_code == 404

    cancel_resp = client.post("/jobs/does-not-exist/cancel")
    assert cancel_resp.status_code == 404
