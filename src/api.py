"""
FastAPI application for Apple Notes Lens.
"""

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional, Union
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.bm25 import BM25Index
from src.chain import Chunker, RAGChain
from src.config import Settings
from src.embedder import Embedder, OllamaEmbedder
from src.llm import LLM, OllamaLLM
from src.models import QAResult
from src.notes_exporter import extract_text_from_html
from src.retriever import HybridRetriever, Retriever
from src.store import ChromaStore, VectorStore
from src.sync import SyncState, incremental_sync_detailed

logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    """Request body for ingest."""

    export_dir: Optional[str] = Field(
        default=None,
        description="Directory containing exported Apple Notes HTML files.",
    )
    reindex: bool = Field(
        default=False,
        description="If true, clears the store and reindexes all notes.",
    )
    mode: Optional[Literal["full", "delta"]] = Field(
        default=None,
        description="Ingest mode. `full` rebuilds everything, `delta` ingests only changes.",
    )


class AskRequest(BaseModel):
    """Request body for ask."""

    query: str = Field(..., description="User query to answer.")


class SearchRequest(BaseModel):
    """Request body for search."""

    query: str = Field(..., description="Search query for notes.")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of results to return (defaults to settings.TOP_K).",
    )


class SearchResponse(BaseModel):
    """Response body for search."""

    query: str
    top_k: int
    results: list[dict]


class IngestJobStatus(BaseModel):
    """Public response model for ingest job state."""

    job_id: str
    status: Literal[
        "queued",
        "running",
        "cancelling",
        "completed",
        "failed",
        "cancelled",
    ]
    phase: Literal["preparing", "scanning", "indexing", "finalizing"]
    current: int = 0
    total: Optional[int] = None
    message: str = ""
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


class IngestCancelledError(Exception):
    """Raised when an ingest job is cancelled."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_export_dir(request_dir: Optional[str], settings: Settings) -> Path:
    if request_dir:
        return Path(request_dir)
    if settings.notes_export_dir:
        return settings.notes_export_dir
    raise HTTPException(status_code=400, detail="export_dir is required")


def _store_health_ok(store: VectorStore) -> bool:
    try:
        _ = store.size()
        return True
    except Exception:
        return False


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _resolve_ingest_mode(request: IngestRequest) -> Literal["full", "delta"]:
    if request.mode in ("full", "delta"):
        return request.mode
    if request.reindex:
        return "full"
    return "delta"


def _reset_index_state(
    store: VectorStore,
    sync_state: SyncState,
    bm25: Optional[BM25Index],
) -> dict[str, int]:
    if isinstance(store, ChromaStore):
        store.reset_collection()
    else:
        store.clear()

    if bm25:
        bm25.reset()
        bm25.save()

    sync_state.note_metadata = {}
    sync_state.last_sync_time = None
    sync_state.save()

    return {
        "chunks": store.size(),
        "notes_indexed": len(sync_state.note_metadata),
    }


def _run_ingest(
    export_dir: Path,
    embedder: Embedder,
    store: VectorStore,
    chunker: Chunker,
    sync_state: SyncState,
    mode: Literal["full", "delta"],
    bm25: BM25Index | None = None,
    progress_callback: Optional[Callable[[str, int, Optional[int], str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    def report(phase: str, current: int, total: Optional[int], message: str) -> None:
        if progress_callback:
            progress_callback(phase, current, total, message)

    def ensure_not_cancelled() -> None:
        if should_cancel and should_cancel():
            raise IngestCancelledError("Ingest cancelled")

    report("scanning", 0, None, "Scanning export folder...")
    changed_notes, removed_note_ids, delta_summary = incremental_sync_detailed(
        export_dir,
        sync_state,
    )
    report(
        "scanning",
        delta_summary.get("scanned_notes", 0),
        delta_summary.get("scanned_notes", 0),
        f"Scanned {delta_summary.get('scanned_notes', 0)} notes.",
    )

    ensure_not_cancelled()

    for note_id in removed_note_ids:
        ensure_not_cancelled()
        store.delete_note(note_id)
        if bm25:
            bm25.remove_note(note_id)

    total_changed = len(changed_notes)
    total_chunks = 0
    report("indexing", 0, total_changed, "Indexing changed notes...")

    for idx, note in enumerate(changed_notes, start=1):
        ensure_not_cancelled()
        if bm25:
            bm25.remove_note(note.id)

        text = extract_text_from_html(note.body)
        chunks = chunker.chunk_text(text, note.id)
        for chunk in chunks:
            ensure_not_cancelled()
            chunk.embedding = embedder.embed(chunk.text)

        if chunks:
            store.add_chunks(chunks)
            total_chunks += len(chunks)
            if bm25:
                bm25.add_documents(chunks)

        report("indexing", idx, total_changed, f"Indexed {idx}/{total_changed} notes.")

    report("finalizing", total_changed, total_changed, "Finalizing index state...")
    if bm25:
        bm25.save()

    ensure_not_cancelled()

    return {
        "mode": mode,
        "changed_notes": len(changed_notes),
        "removed_notes": len(removed_note_ids),
        "chunks_indexed": total_chunks,
        "delta_summary": delta_summary,
        "last_sync_time": (
            sync_state.last_sync_time.isoformat() if sync_state.last_sync_time else None
        ),
    }


def create_app(
    settings: Optional[Settings] = None,
    embedder: Optional[Embedder] = None,
    store: Optional[VectorStore] = None,
    llm: Optional[LLM] = None,
    retriever: Optional[Union[Retriever, HybridRetriever]] = None,
    chain: Optional[RAGChain] = None,
    chunker: Optional[Chunker] = None,
    sync_state: Optional[SyncState] = None,
    bm25_index: Optional[BM25Index] = None,
) -> FastAPI:
    """Create and configure the FastAPI app."""
    settings = settings or Settings()

    embedder = embedder or OllamaEmbedder(
        model_name=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    store = store or ChromaStore(db_path=settings.chroma_db_path, embedder=embedder)
    llm = llm or OllamaLLM(
        model_name=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
    )
    if bm25_index is None:
        bm25 = BM25Index(settings.bm25_index_path)
        bm25.load()
    else:
        bm25 = bm25_index

    retriever = retriever or HybridRetriever(
        store=store,
        embedder=embedder,
        bm25=bm25,
        rerank_enabled=settings.rerank_enabled,
        hybrid_topn=settings.hybrid_topn,
        rerank_candidates_n=settings.rerank_candidates_n,
        final_context_k=settings.final_context_k,
        max_context_tokens=settings.max_context_tokens,
        rerank_cache_ttl_seconds=settings.rerank_cache_ttl_seconds,
        rerank_backend=settings.rerank_backend,
        cross_encoder_model=settings.cross_encoder_model,
        constraint_mode=settings.constraint_mode,
        time_query_require_time_evidence=settings.time_query_require_time_evidence,
        rerank_llm=llm,
    )
    chain = chain or RAGChain(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k,
        min_score=settings.min_similarity_score,
    )
    chunker = chunker or Chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    sync_state = sync_state or SyncState()

    app = FastAPI(title="Apple Notes Lens", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store
    app.state.llm = llm
    app.state.retriever = retriever
    app.state.chain = chain
    app.state.chunker = chunker
    app.state.sync_state = sync_state
    app.state.bm25 = bm25
    app.state.last_ingest_summary = None

    job_lock = threading.Lock()
    ingest_jobs: dict[str, dict] = {}
    ingest_cancel_events: dict[str, threading.Event] = {}

    def get_job_or_404(job_id: str) -> dict:
        with job_lock:
            job = ingest_jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Ingest job not found")
            return dict(job)

    def update_job(job_id: str, **changes: object) -> None:
        with job_lock:
            existing = ingest_jobs.get(job_id)
            if not existing:
                return
            existing.update(changes)

    @app.exception_handler(StarletteHTTPException)
    def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        if exc.status_code == 404:
            return JSONResponse(
                status_code=404,
                content={
                    "detail": "Endpoint not found. Are you running the latest app?",
                    "available_endpoints": [
                        "/ingest",
                        "/jobs/ingest",
                        "/jobs/{job_id}",
                        "/jobs/{job_id}/cancel",
                        "/reset_notes_index",
                        "/ask",
                        "/search",
                        "/health",
                        "/stats",
                    ],
                },
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.on_event("startup")
    def _auto_ingest_on_startup() -> None:
        if not settings.auto_ingest_on_startup:
            return
        if settings.notes_export_dir is None:
            return
        if not settings.notes_export_dir.exists():
            return
        app.state.last_ingest_summary = _run_ingest(
            export_dir=settings.notes_export_dir,
            embedder=embedder,
            store=store,
            chunker=chunker,
            sync_state=sync_state,
            mode="delta",
            bm25=bm25,
        )

    @app.on_event("startup")
    def _warmup_reranker() -> None:
        if not settings.rerank_warmup_enabled:
            return
        if isinstance(retriever, HybridRetriever):

            def _run_warmup() -> None:
                try:
                    warmed = retriever.warmup_reranker()
                    logger.info("Reranker warmup complete (success=%s)", warmed)
                except Exception as err:
                    logger.warning("Reranker warmup failed: %s", err)

            threading.Thread(
                target=_run_warmup,
                name="reranker-warmup",
                daemon=True,
            ).start()

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict:
        export_dir = _resolve_export_dir(request.export_dir, settings)
        mode = _resolve_ingest_mode(request)

        if not export_dir.exists() or not export_dir.is_dir():
            raise HTTPException(status_code=400, detail="export_dir must be a directory")

        if mode == "full":
            _reset_index_state(store=store, sync_state=sync_state, bm25=bm25)

        summary = _run_ingest(
            export_dir=export_dir,
            embedder=embedder,
            store=store,
            chunker=chunker,
            sync_state=sync_state,
            mode=mode,
            bm25=bm25,
        )
        app.state.last_ingest_summary = summary
        return summary

    @app.post("/jobs/ingest")
    def start_ingest_job(request: IngestRequest) -> dict:
        export_dir = _resolve_export_dir(request.export_dir, settings)
        mode = _resolve_ingest_mode(request)

        if not export_dir.exists() or not export_dir.is_dir():
            raise HTTPException(status_code=400, detail="export_dir must be a directory")

        job_id = uuid4().hex
        started_at = _now_iso()
        cancel_event = threading.Event()

        with job_lock:
            ingest_jobs[job_id] = IngestJobStatus(
                job_id=job_id,
                status="queued",
                phase="preparing",
                current=0,
                total=None,
                message="Job queued.",
                started_at=started_at,
                finished_at=None,
                error=None,
                result=None,
            ).model_dump()
            ingest_cancel_events[job_id] = cancel_event

        def run_job() -> None:
            update_job(job_id, status="running", phase="preparing", message="Preparing ingest...")
            try:
                if mode == "full":
                    _reset_index_state(store=store, sync_state=sync_state, bm25=bm25)

                def report(phase: str, current: int, total: Optional[int], message: str) -> None:
                    update_job(
                        job_id,
                        status="running",
                        phase=phase,
                        current=current,
                        total=total,
                        message=message,
                    )

                summary = _run_ingest(
                    export_dir=export_dir,
                    embedder=embedder,
                    store=store,
                    chunker=chunker,
                    sync_state=sync_state,
                    mode=mode,
                    bm25=bm25,
                    progress_callback=report,
                    should_cancel=cancel_event.is_set,
                )
                app.state.last_ingest_summary = summary
                update_job(
                    job_id,
                    status="completed",
                    phase="finalizing",
                    current=summary.get("changed_notes", 0),
                    total=summary.get("changed_notes", 0),
                    message="Ingest completed.",
                    finished_at=_now_iso(),
                    result=summary,
                    error=None,
                )
            except IngestCancelledError:
                update_job(
                    job_id,
                    status="cancelled",
                    phase="finalizing",
                    message="Ingest cancelled.",
                    finished_at=_now_iso(),
                    error=None,
                )
            except Exception as err:
                update_job(
                    job_id,
                    status="failed",
                    phase="finalizing",
                    message="Ingest failed.",
                    finished_at=_now_iso(),
                    error=str(err),
                )

        thread = threading.Thread(target=run_job, name=f"ingest-job-{job_id[:8]}", daemon=True)
        thread.start()

        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def get_ingest_job(job_id: str) -> dict:
        return get_job_or_404(job_id)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_ingest_job(job_id: str) -> dict:
        with job_lock:
            job = ingest_jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Ingest job not found")

            status = job.get("status")
            if status in {"completed", "failed", "cancelled"}:
                return {"job_id": job_id, "status": status}

            cancel_event = ingest_cancel_events.get(job_id)
            if cancel_event:
                cancel_event.set()
            job["status"] = "cancelling"
            job["message"] = "Cancellation requested."
            return {"job_id": job_id, "status": "cancelling"}

    @app.post("/reset_notes_index")
    def reset_notes_index() -> dict:
        summary = _reset_index_state(store=store, sync_state=sync_state, bm25=bm25)
        app.state.last_ingest_summary = None
        return {
            "status": "ok",
            "message": "Notes index reset completed.",
            "summary": summary,
        }

    @app.post("/ask", response_model=QAResult)
    def ask(request: AskRequest) -> QAResult:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        return chain.ask(request.query)

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        top_k = request.top_k or settings.top_k
        citations = retriever.retrieve(request.query, top_k=top_k)
        results = [
            {
                "chunk_id": c.chunk_id,
                "note_id": c.note_id,
                "source": c.source,
                "text": c.text,
                "score": c.score,
            }
            for c in citations
        ]
        return SearchResponse(query=request.query, top_k=top_k, results=results)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "embedder": embedder.health(),
            "llm": llm.health(),
            "store": _store_health_ok(store),
        }

    @app.get("/stats")
    def stats() -> dict:
        last_sync = sync_state.last_sync_time.isoformat() if sync_state.last_sync_time else None
        chroma_bytes = _path_size_bytes(settings.chroma_db_path)
        bm25_bytes = _path_size_bytes(settings.bm25_index_path)
        sync_state_bytes = _path_size_bytes(sync_state.state_file)
        total_store_bytes = chroma_bytes + bm25_bytes + sync_state_bytes
        return {
            "chunks": store.size(),
            "notes_indexed": len(sync_state.note_metadata),
            "last_sync_time": last_sync,
            "last_ingest": app.state.last_ingest_summary,
            "configured_models": {
                "embed_model": settings.ollama_embedding_model,
                "chat_model": settings.ollama_chat_model,
            },
            "delta_summary": (
                app.state.last_ingest_summary.get("delta_summary")
                if isinstance(app.state.last_ingest_summary, dict)
                else None
            ),
            "store_size_bytes": total_store_bytes,
            "storage_breakdown_bytes": {
                "chroma": chroma_bytes,
                "bm25": bm25_bytes,
                "sync_state": sync_state_bytes,
            },
            "manifest_schema_version": sync_state.schema_version,
            "bm25_schema_version": bm25.schema_version if bm25 else None,
            "chroma_db_path": str(settings.chroma_db_path),
            "bm25_index_path": str(settings.bm25_index_path),
            "sync_state_path": str(sync_state.state_file),
        }

    return app
