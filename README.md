# Apple Notes RAG Q&A App

A local-first Q&A application for your Apple Notes using Retrieval Augmented Generation (RAG). Built with Python and powered by Ollama for completely offline inference.

## Features

✅ **Local-First**: All processing happens locally - no cloud APIs or external services  
✅ **Apple Notes Integration**: Export and sync notes directly from Apple Notes app  
✅ **RAG Q&A**: Answer questions about your notes with citations  
✅ **Hybrid Search**: Semantic + lexical (BM25) retrieval  
✅ **Incremental Sync**: Only process changed notes for efficiency  
✅ **Deterministic Testing**: Built with reproducible, testable fake implementations

## Requirements

- **Python**: 3.12+
- **macOS**: For Apple Notes integration (export via AppleScript)
- **Ollama**: For local LLM inference
  - Download: https://ollama.ai
  - Required models:
    - `nomic-embed-text` (embeddings)
    - `neural-chat` (chat/generation)

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure

```bash
# Copy example config
cp .env.example .env

# Edit .env if needed (defaults work for local Ollama)
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_EMBEDDING_MODEL=nomic-embed-text
# OLLAMA_CHAT_MODEL=neural-chat
```

### 3. Start Ollama

```bash
# In a separate terminal, start Ollama
ollama serve

# Pull models (in another terminal)
ollama pull nomic-embed-text
ollama pull neural-chat
```

### 4. Export Apple Notes

```python
from src.notes_exporter import export_notes_interactive

# Opens folder picker for exporting notes from Apple Notes
export_notes_interactive()
```

### 5. Run Tests

```bash
pytest -v
```

### 6. Start the API

```bash
uvicorn main:app --reload
```

If `NOTES_EXPORT_DIR` is set and `AUTO_INGEST_ON_STARTUP=true`, the app will
ingest notes and update the Chroma DB on startup.

### 7. Query the API

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"export_dir": "/path/to/exported_notes"}'

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "What are the key points from my notes?"}'
```

## Usage

### Python API

```python
from pathlib import Path

from src.config import Settings
from src.sync import incremental_sync
from src.embedder import OllamaEmbedder
from src.store import ChromaStore
from src.retriever import Retriever
from src.chain import RAGChain
from src.llm import OllamaLLM

# Load config
config = Settings()

# Sync notes
changed_notes, removed_note_ids = incremental_sync(
    export_dir=Path("path/to/exported_notes")
)

# Setup RAG chain
embedder = OllamaEmbedder(
    model_name=config.ollama_embedding_model,
    base_url=config.ollama_base_url,
)
store = ChromaStore(db_path=config.chroma_db_path, embedder=embedder)
llm = OllamaLLM(
    model_name=config.ollama_chat_model,
    base_url=config.ollama_base_url,
)
retriever = Retriever(store=store, embedder=embedder)
chain = RAGChain(retriever=retriever, llm=llm)

# Ask questions
result = chain.ask("What are the key points from my notes?")
print(f"Answer: {result.answer}")
print(f"Citations: {result.citations}")
print(f"Confidence: {result.confidence}")
```

## Project Structure

```
.
├── scripts/
│   └── export_notes.applescript    # AppleScript for exporting notes
├── src/
│   ├── __init__.py
│   ├── models.py                   # Pydantic data models
│   ├── config.py                   # Settings & configuration
│   ├── notes_exporter.py           # Notes export & parsing
│   ├── sync.py                     # Incremental sync logic
│   ├── embedder.py                 # Embeddings (Ollama + Fake)
│   ├── store.py                    # Vector store (InMemory + Chroma)
│   ├── retriever.py                # Retrieval logic
│   ├── llm.py                      # LLM interface (Ollama + Fake)
│   ├── chain.py                    # RAG chain orchestration
│   └── api.py                      # FastAPI endpoints
├── evals/
│   ├── dataset.json                # Eval dataset
│   ├── fixture_notes.json          # Fixture corpus
│   └── eval.py                     # Recall@k evaluation
├── main.py                         # FastAPI entrypoint
├── tests/
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_sync.py
│   ├── test_embedder.py
│   ├── test_store.py
│   ├── test_retriever.py
│   ├── test_llm.py
│   ├── test_chain.py
│   └── test_api.py
├── data/
│   ├── sync_state.json             # Sync state (git-ignored)
│   └── chroma_db/                  # Vector database (git-ignored)
├── .env.example                    # Example config (no secrets)
├── pyproject.toml
├── README.md
└── plan.md                         # Implementation plan
```

## Development

### Running Tests

```bash
# All tests
pytest -v

# Specific test file
pytest tests/test_chain.py -v

# With coverage
pytest --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Evaluation

```bash
python evals/eval.py
```

For real embeddings, run Ollama and pass:

```bash
python evals/eval.py --use-ollama --use-chroma
```

If you change embedding models, reindex the vector store (collection dimensions are fixed):

```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"export_dir": "/path/to/exported_notes", "reindex": true}'
```

### Test Strategy

All tests use **deterministic fake implementations**:

- `FakeEmbedder`: Hash-based embeddings (reproducible, no Ollama needed)
- `FakeLLM`: Canned responses with customization
- `InMemoryStore`: Simple dict-based vector store

This means **tests run without Ollama** and are completely reproducible in CI/CD.

## Architecture

### Data Flow

```
Apple Notes → Export → Parse HTML → Sync State
    ↓
Split into Chunks → Embed (Ollama) → Vector Store (Chroma)
    ↓
Tokenize → BM25 Index
    ↓
Query → Hybrid Retrieve (BM25 + Vector) → Build Prompt → LLM (Ollama)
    ↓
Parse Citations → QAResult (answer, citations, confidence)
```

### Key Components

| Component       | Purpose                   | Implementations                                      |
| --------------- | ------------------------- | ---------------------------------------------------- |
| **Embedder**    | Convert text to vectors   | OllamaEmbedder, FakeEmbedder                         |
| **VectorStore** | Store & search embeddings | ChromaStore, InMemoryStore                           |
| **Retriever**   | Semantic search wrapper   | Single class (wraps Embedder + Store)                |
| **LLM**         | Text generation           | OllamaLLM, FakeLLM                                   |
| **RAGChain**    | Orchestrates Q&A pipeline | retrieve → build prompt → generate → parse citations |

## Configuration

Create `.env` from `.env.example`:

```dotenv
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434          # Ollama server
OLLAMA_EMBEDDING_MODEL=nomic-embed-text         # Embedding model
OLLAMA_CHAT_MODEL=neural-chat                   # Chat/generation model

# Vector Store Configuration
CHROMA_DB_PATH=data/chroma_db                   # Persistent vector DB

# RAG Configuration
CHUNK_SIZE=512                                  # Characters per chunk
CHUNK_OVERLAP=50                                # Overlap between chunks
TOP_K=5                                         # Chunks to retrieve per query
MIN_SIMILARITY_SCORE=0.8                        # Minimum similarity for snippets
BM25_INDEX_PATH=data/bm25_index.json            # BM25 index persistence

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=False

# Logging & Debug
DEBUG=False

# Notes Export
NOTES_EXPORT_DIR=                               # Optional: path to exported notes
AUTO_INGEST_ON_STARTUP=false                    # Set true to ingest on app start
```

## FAQ

**Q: Do I need an internet connection?**  
A: No! All processing is local. You only need Ollama running locally.

**Q: Can I use this with a cloud LLM provider?**  
A: Not currently - the design prioritizes local-first. You could extend it by implementing custom `Embedder` and `LLM` classes.

**Q: What about privacy?**  
A: Your notes never leave your machine. Everything runs locally on your device.

**Q: Can I use this on non-macOS?**  
A: The Apple Notes export requires macOS, but you could manually export notes as HTML and use the rest of the system.

## Testing Results

All C4 implementation tests passing:

```
✅ test_retriever.py        (9 tests)
✅ test_llm.py              (18 tests)
✅ test_chain.py            (25+ tests)
✅ test_models.py           (25 tests)
✅ test_config.py           (varies)
✅ test_sync.py             (34 tests)
✅ test_embedder.py         (20+ tests)
✅ test_store.py            (25+ tests)
━━━━━━━━━━━━━━━━━━━━━━━━
Total: 150+ tests passing
```

## Roadmap

- **C5**: FastAPI REST endpoints
- **C6**: Evaluation suite with metrics
- **C7**: Docker & deployment
- **Future**: Web UI, chat history, note sync automation

## License

MIT

## Contributing

Contributions welcome! Please ensure:

- All tests pass (`pytest`)
- Code is formatted (`black`)
- No linting errors (`ruff`)
- Type hints are present (`mypy`)

---

**Built with**: Python 3.12, Ollama, Chroma, Pydantic, FastAPI
