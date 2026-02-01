"""Pytest configuration"""

import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.chain import Chunker, RAGChain
from src.embedder import FakeEmbedder
from src.llm import FakeLLM
from src.models import Note
from src.notes_exporter import extract_text_from_html
from src.retriever import Retriever
from src.store import InMemoryStore


@pytest.fixture()
def fixture_notes() -> list[Note]:
    """Small corpus of notes for integration-style tests."""
    base_time = datetime(2024, 1, 15, 10, 30, 0)
    return [
        Note(
            id="n1",
            title="Travel Plans",
            body="<html><body>Paris trip in May. Book flights and hotel.</body></html>",
            folder="Notes/Personal",
            created_at=base_time,
            modified_at=base_time,
        ),
        Note(
            id="n2",
            title="Meeting Notes",
            body="<html><body>Discussed Q1 OKRs and hiring roadmap.</body></html>",
            folder="Notes/Work",
            created_at=base_time,
            modified_at=base_time,
        ),
        Note(
            id="n3",
            title="Recipe",
            body="<html><body>Pasta: boil water, add salt, cook for 9 minutes.</body></html>",
            folder="Notes/Home",
            created_at=base_time,
            modified_at=base_time,
        ),
    ]


@pytest.fixture()
def fixture_store(fixture_notes: list[Note]) -> InMemoryStore:
    """InMemoryStore pre-populated with chunks for fixture notes."""
    embedder = FakeEmbedder(dimension=8)
    store = InMemoryStore()
    chunker = Chunker(chunk_size=200, chunk_overlap=0)

    for note in fixture_notes:
        text = extract_text_from_html(note.body)
        chunks = chunker.chunk_text(text, note.id)
        for chunk in chunks:
            chunk.embedding = embedder.embed(chunk.text)
        store.add_chunks(chunks)

    return store


@pytest.fixture()
def fixture_rag_chain(fixture_store: InMemoryStore) -> RAGChain:
    """RAGChain using fake components for deterministic tests."""
    embedder = FakeEmbedder(dimension=8)
    retriever = Retriever(store=fixture_store, embedder=embedder)
    llm = FakeLLM()
    return RAGChain(retriever=retriever, llm=llm, top_k=3)
