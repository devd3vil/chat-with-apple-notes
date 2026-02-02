"""
FastAPI application for Apple Notes RAG.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from src.chain import Chunker, RAGChain
from src.config import Settings
from src.embedder import Embedder, OllamaEmbedder
from src.llm import LLM, OllamaLLM
from src.models import QAResult
from src.notes_exporter import extract_text_from_html
from src.retriever import Retriever
from src.store import ChromaStore, VectorStore
from src.sync import SyncState, incremental_sync


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

def _run_ingest(
    export_dir: Path,
    embedder: Embedder,
    store: VectorStore,
    chunker: Chunker,
    sync_state: SyncState,
) -> dict:
    changed_notes, removed_note_ids = incremental_sync(export_dir, sync_state)

    for note_id in removed_note_ids:
        store.delete_note(note_id)

    total_chunks = 0
    for note in changed_notes:
        text = extract_text_from_html(note.body)
        chunks = chunker.chunk_text(text, note.id)
        for chunk in chunks:
            chunk.embedding = embedder.embed(chunk.text)
        if chunks:
            store.add_chunks(chunks)
            total_chunks += len(chunks)

    return {
        "changed_notes": len(changed_notes),
        "removed_notes": len(removed_note_ids),
        "chunks_indexed": total_chunks,
    }


def create_app(
    settings: Optional[Settings] = None,
    embedder: Optional[Embedder] = None,
    store: Optional[VectorStore] = None,
    llm: Optional[LLM] = None,
    retriever: Optional[Retriever] = None,
    chain: Optional[RAGChain] = None,
    chunker: Optional[Chunker] = None,
    sync_state: Optional[SyncState] = None,
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
    retriever = retriever or Retriever(store=store, embedder=embedder)
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

    app = FastAPI(title="Apple Notes RAG", version="0.1.0")
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store
    app.state.llm = llm
    app.state.retriever = retriever
    app.state.chain = chain
    app.state.chunker = chunker
    app.state.sync_state = sync_state

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
        _run_ingest(
            export_dir=settings.notes_export_dir,
            embedder=embedder,
            store=store,
            chunker=chunker,
            sync_state=sync_state,
        )

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict:
        export_dir = _resolve_export_dir(request.export_dir, settings)

        if not export_dir.exists() or not export_dir.is_dir():
            raise HTTPException(status_code=400, detail="export_dir must be a directory")

        if request.reindex:
            if isinstance(store, ChromaStore):
                store.reset_collection()
            else:
                store.clear()
            sync_state.note_metadata = {}
            sync_state.last_sync_time = None
            sync_state.save()

        return _run_ingest(
            export_dir=export_dir,
            embedder=embedder,
            store=store,
            chunker=chunker,
            sync_state=sync_state,
        )

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
            {"chunk_id": c.chunk_id, "text": c.text, "score": c.score}
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
        return {
            "chunks": store.size(),
            "last_sync_time": last_sync,
            "chroma_db_path": str(settings.chroma_db_path),
        }

    return app
