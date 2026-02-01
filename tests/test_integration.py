"""
High-level integration test using a golden dataset.
"""

import json
from datetime import datetime
from pathlib import Path

from src.chain import Chunker, RAGChain
from src.embedder import FakeEmbedder
from src.llm import FakeLLM
from src.models import Note
from src.retriever import Retriever
from src.store import InMemoryStore


def _load_golden_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_golden_set_integration() -> None:
    golden_path = Path(__file__).parent / "golden_set.json"
    golden_set = _load_golden_set(golden_path)

    embedder = FakeEmbedder(dimension=16)
    store = InMemoryStore()
    retriever = Retriever(store=store, embedder=embedder)
    llm = FakeLLM()
    llm.set_response("Answer [1]")
    chain = RAGChain(retriever=retriever, llm=llm, top_k=3)
    chunker = Chunker(chunk_size=1024, chunk_overlap=0)

    base_time = datetime(2024, 1, 15, 10, 30, 0)
    notes: list[Note] = []
    for item in golden_set:
        notes.append(
            Note(
                id=item["note_id"],
                title=item["title"],
                body=item["body"],
                folder="Notes/Golden",
                created_at=base_time,
                modified_at=base_time,
            )
        )

    for note in notes:
        chunks = chunker.chunk_text(note.body, note.id)
        for chunk in chunks:
            chunk.embedding = embedder.embed(chunk.text)
        store.add_chunks(chunks)

    for item in golden_set:
        result = chain.ask(item["query"])
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == item["expected_chunk_id"]
        assert 0.0 <= result.confidence <= 1.0
