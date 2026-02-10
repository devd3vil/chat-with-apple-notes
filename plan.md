# RAG Q&A over Apple Notes - Project Plan

**Goal:** Build a local-first Q&A system over Apple Notes using Ollama embeddings, Chroma vector store, and FastAPI.

**Tech Stack:**

- Python 3.12+ FastAPI
- Chroma (local vector database)
- Ollama (embeddings + chat models)
- AppleScript (incremental note sync)

**Constraints:**

- ✅ Local-first: no external APIs, all data stays on machine
- ✅ Incremental sync: track note changes via id, modification_date, body
- ✅ Citations: top-k snippets returned; LLM instructed to cite like `[1][2]`
- ✅ Deterministic tests: FakeEmbedder for unit tests
- ✅ Safe defaults: .env.example, .env in .gitignore

---

## File Tree

```
chatwithNotes/
├── scripts/
│   └── export_notes.applescript          # Note exporter (existing)
├── src/
│   ├── __init__.py
│   ├── notes_exporter.py                 # AppleScript wrapper (existing)
│   ├── models.py                         # Pydantic models (Note, Chunk, QAResult)
│   ├── sync.py                           # Incremental sync logic
│   ├── embedder.py                       # Ollama embeddings wrapper
│   ├── store.py                          # Chroma vector store wrapper
│   ├── retriever.py                      # RAG retriever (top-k)
│   ├── llm.py                            # Ollama chat wrapper
│   ├── chain.py                          # RAG chain (retrieve → cite → ask)
│   ├── api.py                            # FastAPI endpoints
│   └── config.py                         # Config & env loading
├── tests/
│   ├── conftest.py                       # Pytest fixtures
│   ├── test_models.py
│   ├── test_embedder.py                  # Unit: FakeEmbedder
│   ├── test_store.py                     # Unit: in-memory store
│   ├── test_retriever.py                 # Unit: retrieval logic
│   ├── test_sync.py                      # Unit: sync logic
│   ├── test_chain.py                     # Integration: RAG chain
│   └── test_api.py                       # Integration: API endpoints
├── evals/
│   ├── dataset.json                      # 20 Q&A pairs (fixture)
│   ├── eval.py                           # Recall@k computation
│   └── results.jsonl                     # Eval results
├── data/
│   └── (gitignored) chroma_db/           # Local Chroma store
├── .env.example
├── .env                                  # (gitignored)
├── .gitignore                            # (existing)
├── pyproject.toml                        # (update with new deps)
├── README.md                             # (update)
├── plan.md                               # (this file)
├── docker-compose.yml                    # (C6: Ollama + app)
└── Dockerfile                            # (C6: FastAPI container)
```

---

## Git commits

After each milestone or checklist make sure you commit to git

## Milestones & Checklists

### C0: Project Setup ✅ (DONE)

- [x] AppleScript exporter
- [x] Python wrapper
- [x] pyproject.toml, README, .env.example
- [x] .gitignore

---

### C1: Models & Configuration ✅ (DONE)

**Goal:** Define data structures and config loading

**Tasks:**

- [x] Create `src/models.py`:
  - `Note(id, title, body, folder, created_at, modified_at)`
  - `Chunk(note_id, text, chunk_idx, start_char, end_char, embedding=None)`
  - `QAResult(query, answer, citations: list[Citation], confidence)`
  - `Citation(chunk_id, text, score)`
- [x] Create `src/config.py`:
  - Load from `.env`: `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_CHAT_MODEL`, `CHROMA_DB_PATH`, `CHUNK_SIZE`, `TOP_K`
  - Defaults: embedding=`nomic-embed-text`, chat=`neural-chat`, chunk_size=512, top_k=5
- [x] Update `pyproject.toml`: add `pydantic-settings`
- [x] Tests: `test_models.py` (25 tests), `test_config.py` (validation + env loading)

**Files to Create:**

- `src/models.py`
- `src/config.py`
- `tests/test_models.py`
- `tests/test_config.py`

---

### C2: Incremental Sync ✅ (DONE)

**Goal:** Track changes in Apple Notes and only re-sync modified notes

**Tasks:**

- [x] Create `src/sync.py`:
  - `SyncState(last_sync_time, note_metadata: dict[note_id -> (modified_at, hash)])`
  - `incremental_sync(export_dir)` → list of changed Note objects
  - Compare: note id, modification_date, body hash
  - Persist SyncState to `data/sync_state.json` (gitignored)
- [x] Extend `src/notes_exporter.py` to parse HTML exports:
  - `parse_exported_html(html_file) → Note`
  - Extract title from filename `YYYYMMDD <title> [<id>].html`
  - Parse body from HTML
  - `extract_text_from_html()` for text extraction
- [x] Tests: 18 sync tests + 16 exporter tests (34 total); full cycle: new → unchanged → modified → removed

**Files to Create/Update:**

- `src/sync.py`
- `src/notes_exporter.py` (add parser)
- `tests/test_sync.py`
- `data/sync_state.json` (gitignored, auto-created)

---

### C3: Vector Store & Embeddings ✅ (DONE)

**Goal:** Abstract Chroma and Ollama; enable testing with fakes

**Tasks:**

- [x] Create `src/embedder.py`:
  - Abstract `Embedder` interface (embed text → embedding vector)
  - `OllamaEmbedder(model_name, base_url)` (calls Ollama API)
  - `FakeEmbedder` (deterministic: hash-based embeddings for tests)
  - Health check: `embedder.health() → bool`
- [x] Create `src/store.py`:
  - Abstract `VectorStore` interface:
    - `add_chunks(chunks: list[Chunk]) → None`
    - `search(query_embedding, top_k) → list[(chunk_id, chunk_text, score)]`
    - `delete_note(note_id) → None`
  - `ChromaStore(db_path, embedder)` (wraps Chroma)
  - `InMemoryStore(embedder)` (for unit tests; cosine similarity)
- [x] Tests: 25 embedder tests + 20 store tests (45 total)

**Files to Create:**

- `src/embedder.py`
- `src/store.py`
- `tests/test_embedder.py`
- `tests/test_store.py`

---

### C4: Retriever & RAG Chain

**Goal:** Retrieve top-k chunks and generate cited answers

**Tasks:**

- [ ] Create `src/retriever.py`:
  - `Retriever(store)` class
  - `retrieve(query_text: str, top_k: int) → list[Citation]` (embed query, search, return with scores)
- [ ] Create `src/llm.py`:
  - Abstract `LLM` interface: `generate(prompt, max_tokens) → str`
  - `OllamaLLM(model_name, base_url)` (calls Ollama API)
  - `FakeLLM` (returns canned response for tests)
  - Health check: `llm.health() → bool`
- [ ] Create `src/chain.py`:
  - `RAGChain(retriever, llm)` class
  - `ask(query: str) → QAResult`:
    1. Retrieve top-k chunks
    2. Build prompt: "Answer based on these snippets. Cite them like [1][2]. <snippets> Question: {query}"
    3. Generate answer with LLM
    4. Parse citations from answer
    5. Return `QAResult(query, answer, citations, confidence)`
- [ ] Tests:
  - `test_retriever.py`: mock store, check ranking
  - `test_chain.py`: integration test with FakeEmbedder + FakeLLM + InMemoryStore

**Files to Create:**

- `src/retriever.py`
- `src/llm.py`
- `src/chain.py`
- `tests/test_retriever.py`
- `tests/test_chain.py`

---

### C5: FastAPI Server & Ingest Pipeline

**Goal:** Expose sync + RAG via REST API

**Tasks:**

- [ ] Create `src/api.py`:
  - `FastAPI()` app
  - `POST /ingest`: trigger sync, chunk, embed, store (return summary)
  - `POST /ask`: query the RAG chain (return QAResult with citations)
  - `GET /health`: check Ollama, Chroma, store status
  - `GET /stats`: store size, last sync time, index stats
- [ ] Create chunking logic in `src/chain.py` or new `src/chunker.py`:
  - Overlap-aware chunking (512-char chunks, 50-char overlap)
  - `chunk_note(note) → list[Chunk]`
- [ ] Create `main.py` or app entry:
  - Initialize config, embedder, store, chain
  - Setup logging
  - Health checks at startup
- [ ] Tests:
  - `test_api.py`: mock chain, test endpoints, status codes

**Files to Create/Update:**

- `src/api.py`
- `src/chunker.py` (optional)
- `main.py` (or `src/app.py`)
- `tests/test_api.py`

---

### C6: Evaluation & Testing Suite

**Goal:** Unit + integration tests; eval script for Recall@k

**Tasks:**

- [ ] Create `evals/dataset.json`:
  - 20 Q&A pairs (hand-curated or generated from fixture notes)
  - Format: `[{"query": "...", "expected_chunks": ["chunk_id_1", ...]}, ...]`
- [ ] Create `evals/eval.py`:
  - Load fixture corpus into temp Chroma
  - For each Q in dataset: retrieve top-k, check if expected chunks present
  - Compute Recall@1, Recall@3, Recall@5
  - Output: `Recall@1: 0.85, Recall@3: 0.95, Recall@5: 1.0`
- [ ] Update `tests/conftest.py`:
  - Fixture: `fixture_notes()` → 10 test notes
  - Fixture: `fixture_store()` → populated InMemoryStore
  - Fixture: `fixture_rag_chain()` → chain with fakes
  - Skip real Ollama tests if unavailable (use `pytest.mark.skipif`)
- [ ] Create `evals/fixture_notes.json`: small corpus for testing
- [ ] Update `pyproject.toml`:
  - Add test scripts: `pytest`, `pytest --cov`, eval runner
  - Add dev deps: `chromadb`, `ollama-python`
- [ ] Tests checklist:
  - [x] Unit: models, config, embedder, store, retriever
  - [ ] Integration: sync, chain, API (with fakes)
  - [ ] Eval: Recall@k on fixture corpus
  - [ ] Optional real Ollama: marker to skip if unavailable

**Files to Create:**

- `evals/dataset.json`
- `evals/fixture_notes.json`
- `evals/eval.py`
- `tests/conftest.py` (update)
- `pyproject.toml` (update)

---

### C8: Make sure you write high level integration test

1. create a golden test set
2. Write high level integration test

### C9: Docker & Deployment (Optional)

**Goal:** Bundle app + Ollama for easy deployment

**Tasks:**

- [ ] Create `docker-compose.yml`:
  - Service: `ollama` (image: `ollama:latest`, volumes for model cache)
  - Service: `app` (build from Dockerfile, depends_on ollama)
  - Expose: port 8000 (FastAPI), shared network
- [ ] Create `Dockerfile`:
  - Python 3.12 base image
  - COPY source
  - RUN pip install -e .
  - CMD uvicorn main:app --host 0.0.0.0 --port 8000
- [ ] Update `.env.example`:
  - `OLLAMA_BASE_URL=http://ollama:11434` (for Docker)
  - `CHROMA_DB_PATH=/app/data/chroma_db`
- [ ] Tests: integration test with Docker compose (optional; skip in CI if unavailable)

**Files to Create:**

- `docker-compose.yml`
- `Dockerfile`
- `.env.example` (update)

---

## Dependencies to Add

Update `pyproject.toml`:

```toml
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.104",
    "uvicorn>=0.24",
    "chromadb>=0.4.0",
    "ollama>=0.1.0",  # Ollama Python client
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "pre-commit>=3.0",
    "httpx>=0.25",  # For API testing
]
```

---

## Safety Checklist

- [ ] `.env` in `.gitignore` (no secrets committed)
- [ ] `.env.example` with safe defaults documented
- [ ] All Ollama calls wrapped in try/except with clear error messages
- [ ] Temp directories (tests) cleaned up in fixtures
- [ ] Chroma DB path must be configurable and not hardcoded
- [ ] All user queries sanitized before passing to LLM
- [ ] Rate limiting on API endpoints (optional but recommended)
- [ ] Logging: no sensitive data in logs

---

## Getting Started

```bash
# 1. Install (already done)
pip install -e ".[dev]"

# 2. Start Ollama
ollama serve

# 3. In another terminal, pull models
ollama pull nomic-embed-text
ollama pull neural-chat

# 4. Run tests
pytest

# 5. Start app
uvicorn main:app --reload

# 6. Test API
curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/ask -d '{"query": "..."}'

# 7. Eval
python evals/eval.py
```

---

## Timeline

- **Week 1:** C1 + C2 (models, sync)
- **Week 2:** C3 + C4 (embedder, store, retriever, chain)
- **Week 3:** C5 + C6 (API, eval, tests)
- **Week 4:** C7 (Docker), polish, docs

---

## Success Criteria

✅ Export notes → parse HTML → track changes
✅ Embed chunks with Ollama (deterministic tests with FakeEmbedder)
✅ Store in Chroma locally; retrieve top-k
✅ RAG chain generates cited answers `[1][2]`
✅ FastAPI server with `/ingest`, `/ask`, `/health`
✅ Unit tests (100% coverage of core logic) + integration tests
✅ Eval: Recall@k on 20-Q dataset
✅ All tests pass; no external API calls
✅ Docker compose to run locally

## Ideas for later

1. Create a user interface that can export the delta notes from apple notes.
2. Run the sync notes to the vector db
3. I should be able to search notes through semantic search and also summarize a note
4. Do Q&A with notes and have it saved as history.
