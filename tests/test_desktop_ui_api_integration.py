"""
Desktop UI <-> API integration tests.

These tests focus on the exact API contracts and workflows consumed by
`desktop/ui/app.js`:
- `/health` gating before chat/setup
- `/jobs/ingest` polling lifecycle
- `/ask` response shape for transcript + citations
- cancellation contract used by UI cancel buttons
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api import create_app
from src.chain import Chunker
from src.config import Settings
from src.embedder import Embedder
from src.llm import FakeLLM
from src.store import InMemoryStore
from src.sync import SyncState


def _write_note(export_dir: Path, date: str, title: str, note_id: str, body: str) -> None:
    filename = f"{date} {title} [{note_id}].html"
    html = f"<html><body>{body}</body></html>"
    (export_dir / filename).write_text(html, encoding="utf-8")


class KeywordEmbedder(Embedder):
    """Deterministic lexical embedder for stable integration tests."""

    _TOKENS = (
        "anika",
        "aurora",
        "saturn",
        "december",
        "goals",
        "meeting",
    )

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        lowered = text.lower()
        return [1.0 if token in lowered else 0.0 for token in self._TOKENS]

    def health(self) -> bool:
        return True

    def get_embedding_dim(self) -> int:
        return len(self._TOKENS)


class SlowKeywordEmbedder(KeywordEmbedder):
    """Like KeywordEmbedder but intentionally slow to exercise cancel flow."""

    def embed(self, text: str) -> list[float]:
        time.sleep(0.004)
        return super().embed(text)


def _make_ui_client(
    tmp_path: Path,
    *,
    embedder: Embedder | None = None,
    llm: FakeLLM | None = None,
) -> TestClient:
    settings = Settings(
        chroma_db_path=tmp_path / "chroma_db",
        bm25_index_path=tmp_path / "bm25_index.json",
        ollama_embedding_model="nomic-embed-text",
        ollama_chat_model="neural-chat",
        rerank_enabled=False,
        rerank_warmup_enabled=False,
        min_similarity_score=0.8,
    )
    embedder = embedder or KeywordEmbedder()
    llm = llm or FakeLLM("Answer from local notes [1]")
    chunker = Chunker(chunk_size=200, chunk_overlap=0)
    sync_state = SyncState(state_file=tmp_path / "sync_state.json")
    app = create_app(
        settings=settings,
        embedder=embedder,
        store=InMemoryStore(),
        llm=llm,
        chunker=chunker,
        sync_state=sync_state,
    )
    return TestClient(app)


def _wait_for_terminal_job(client: TestClient, job_id: str, timeout_s: float = 8.0) -> dict:
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


def _assert_job_payload_shape(payload: dict) -> None:
    required = {
        "job_id",
        "status",
        "phase",
        "current",
        "total",
        "message",
        "started_at",
        "finished_at",
        "error",
        "result",
    }
    assert required.issubset(payload.keys())
    assert payload["status"] in {
        "queued",
        "running",
        "cancelling",
        "completed",
        "failed",
        "cancelled",
    }
    assert payload["phase"] in {"preparing", "scanning", "indexing", "finalizing"}
    assert isinstance(payload["current"], int)
    assert payload["total"] is None or isinstance(payload["total"], int)
    assert isinstance(payload["message"], str)


def test_ui_chat_flow_returns_renderable_citations(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_note(
        export_dir,
        "20241201",
        "Project Saturn",
        "p1",
        "anika saturn december goals",
    )
    _write_note(export_dir, "20241202", "Misc", "p2", "meeting aurora")

    llm = FakeLLM("Anika is linked to Project Saturn [1]")
    client = _make_ui_client(tmp_path, llm=llm)

    started = client.post("/jobs/ingest", json={"export_dir": str(export_dir), "mode": "full"})
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    terminal = _wait_for_terminal_job(client, job_id)
    _assert_job_payload_shape(terminal)
    assert terminal["status"] == "completed"

    response = client.post("/ask", json={"query": "who is anika saturn"})
    assert response.status_code == 200
    payload = response.json()

    assert isinstance(payload["answer"], str) and payload["answer"]
    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["citations"], list) and payload["citations"]

    first = payload["citations"][0]
    assert isinstance(first["chunk_id"], str) and first["chunk_id"]
    assert first["note_id"] == "p1"
    assert isinstance(first["text"], str) and "anika" in first["text"].lower()
    assert isinstance(first["score"], float)
    assert first["source"]["note_id"] == "p1"


def test_ui_health_contract_matches_chat_gating(tmp_path: Path) -> None:
    client = _make_ui_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ok"
    assert isinstance(payload["embedder"], bool)
    assert isinstance(payload["llm"], bool)
    assert isinstance(payload["store"], bool)


def test_ui_delta_sync_flow_updates_indexed_content(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    note_path = export_dir / "20241201 Project Saturn [p1].html"
    note_path.write_text("<html><body>aurora saturn december goals</body></html>", encoding="utf-8")

    llm = FakeLLM("Updated owner is Anika [1]")
    client = _make_ui_client(tmp_path, llm=llm)

    full_started = client.post("/jobs/ingest", json={"export_dir": str(export_dir), "mode": "full"})
    assert full_started.status_code == 200
    full_terminal = _wait_for_terminal_job(client, full_started.json()["job_id"])
    assert full_terminal["status"] == "completed"

    note_path.write_text(
        "<html><body>anika saturn december goals</body></html>",
        encoding="utf-8",
    )
    delta_started = client.post(
        "/jobs/ingest",
        json={"export_dir": str(export_dir), "mode": "delta"},
    )
    assert delta_started.status_code == 200
    delta_terminal = _wait_for_terminal_job(client, delta_started.json()["job_id"])
    _assert_job_payload_shape(delta_terminal)
    assert delta_terminal["status"] == "completed"
    assert delta_terminal["result"]["mode"] == "delta"
    assert delta_terminal["result"]["delta_summary"]["updated_notes"] == 1

    ask = client.post("/ask", json={"query": "anika saturn"})
    assert ask.status_code == 200
    citations = ask.json()["citations"]
    assert citations
    assert "anika" in citations[0]["text"].lower()

    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["last_sync_time"] is not None


def test_ui_ingest_cancel_contract(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    body = "anika saturn december goals " * 40
    for idx in range(1, 36):
        _write_note(export_dir, f"202412{idx:02d}", f"Note {idx}", f"p{idx}", body)

    client = _make_ui_client(tmp_path, embedder=SlowKeywordEmbedder())

    started = client.post("/jobs/ingest", json={"export_dir": str(export_dir), "mode": "full"})
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    cancelled = client.post(f"/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    cancel_payload = cancelled.json()
    assert cancel_payload["job_id"] == job_id
    assert cancel_payload["status"] in {"cancelling", "cancelled"}

    terminal = _wait_for_terminal_job(client, job_id, timeout_s=10.0)
    _assert_job_payload_shape(terminal)
    assert terminal["status"] == "cancelled"


def test_ui_ask_error_payload_for_empty_query(tmp_path: Path) -> None:
    client = _make_ui_client(tmp_path)
    response = client.post("/ask", json={"query": "   "})
    assert response.status_code == 400
    payload = response.json()
    assert isinstance(payload["detail"], str)
    assert payload["detail"] == "query must not be empty"
