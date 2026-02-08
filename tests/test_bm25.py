"""Tests for BM25 index."""

from pathlib import Path
import json

from src.bm25 import BM25Index, tokenize
from src.models import Chunk


def test_tokenize_basic() -> None:
    tokens = tokenize("Hello, World! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "this" not in tokens  # stopword removed


def test_bm25_add_query_and_persist(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.json"
    bm25 = BM25Index(index_path)

    chunks = [
        Chunk(
            id="c1_0",
            note_id="c1",
            text="Estes Park itinerary and riverwalk.",
            chunk_idx=0,
            start_char=0,
            end_char=35,
            embedding=None,
        ),
        Chunk(
            id="c2_0",
            note_id="c2",
            text="Colorado Springs Pikes Peak.",
            chunk_idx=0,
            start_char=0,
            end_char=30,
            embedding=None,
        ),
    ]
    bm25.add_documents(chunks)

    results = bm25.query("Estes Park", top_k=2)
    assert results
    assert results[0][0] == "c1_0"

    bm25.save()
    bm25_loaded = BM25Index(index_path)
    bm25_loaded.load()

    loaded_results = bm25_loaded.query("Estes Park", top_k=2)
    assert loaded_results
    assert loaded_results[0][0] == "c1_0"


def test_bm25_load_migrates_legacy_schema(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.json"
    legacy_payload = {
        "chunk_ids": ["c1_0"],
        "documents": ["legacy doc"],
        "note_ids": ["c1"],
        "tokenized_corpus": [["legacy", "doc"]],
    }
    index_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    bm25 = BM25Index(index_path)
    bm25.load()

    assert bm25.schema_version >= 2
    saved = json.loads(index_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] >= 2
