"""
Evaluate retrieval recall@k on a fixture dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.chain import Chunker
from src.embedder import FakeEmbedder, OllamaEmbedder
from src.models import Note
from src.retriever import HybridRetriever, Retriever
from src.bm25 import BM25Index
from src.store import ChromaStore, InMemoryStore, VectorStore


def _load_notes(path: Path) -> list[Note]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Note.model_validate(item) for item in data]


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_store(
    notes: list[Note],
    chunker: Chunker,
    embedder,
    use_chroma: bool,
) -> VectorStore:
    if use_chroma:
        try:
            temp_dir = TemporaryDirectory()
            store = ChromaStore(db_path=Path(temp_dir.name), embedder=embedder)
            store._temp_dir = temp_dir  # keep alive for duration
        except Exception:
            store = InMemoryStore()
    else:
        store = InMemoryStore()

    for note in notes:
        chunks = chunker.chunk_text(note.body, note.id)
        for chunk in chunks:
            chunk.embedding = embedder.embed(chunk.text)
        store.add_chunks(chunks)

    return store


def _recall_at_k(expected: set[str], retrieved: list[str], k: int) -> int:
    top_k = set(retrieved[:k])
    return 1 if expected.intersection(top_k) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recall@k for retrieval.")
    parser.add_argument("--dataset", default="evals/dataset.json")
    parser.add_argument("--notes", default="evals/dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--ollama-embedding-model", default="nomic-embed-text")
    parser.add_argument("--use-chroma", action="store_true")
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-misses", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    notes_path = Path(args.notes)
    dataset_path = Path(args.dataset)

    notes = _load_notes(notes_path)
    dataset = _load_dataset(dataset_path)

    if args.use_ollama:
        embedder = OllamaEmbedder(
            model_name=args.ollama_embedding_model,
            base_url=args.ollama_base_url,
        )
    else:
        embedder = FakeEmbedder(dimension=64)

    chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    store = _build_store(notes, chunker, embedder, use_chroma=args.use_chroma)
    if args.hybrid:
        bm25 = BM25Index(Path("data/bm25_eval_index.json"))
        for note in notes:
            chunks = chunker.chunk_text(note.body, note.id)
            bm25.add_documents(chunks)
        retriever = HybridRetriever(store=store, embedder=embedder, bm25=bm25)
    else:
        retriever = Retriever(store=store, embedder=embedder)

    totals = {1: 0, 3: 0, 5: 0}
    results = []

    for item in dataset:
        query = item["query"]
        expected = set(item["expected_chunks"])
        trace = None
        if args.trace_misses and hasattr(retriever, "retrieve_with_trace"):
            citations, trace = retriever.retrieve_with_trace(query, top_k=args.top_k)
        else:
            citations = retriever.retrieve(query, top_k=args.top_k)
        retrieved_ids = [c.chunk_id for c in citations]

        for k in totals:
            totals[k] += _recall_at_k(expected, retrieved_ids, k)

        results.append(
            {
                "query": query,
                "expected_chunks": list(expected),
                "retrieved_chunks": retrieved_ids,
            }
        )

        if args.trace_misses and trace and not expected.intersection(retrieved_ids):
            print("\n=== Retrieval Trace (miss) ===")
            print(f"Query: {trace['query']}")
            print(f"Weights: w_vec={trace['w_vec']} w_lex={trace['w_lex']} k_rrf={trace['k_rrf']}")
            print(f"Lex query: {trace['lex_query']}")
            print("Top 10 Lexical:")
            for row in trace["bm25_top"]:
                print(row)
            print("Top 10 Vector:")
            for row in trace["vec_top"]:
                print(row)
            print("Top 10 Fused:")
            for row in trace["fused_top"][:10]:
                print(row)

    total_queries = len(dataset) if dataset else 1
    recall_1 = totals[1] / total_queries
    recall_3 = totals[3] / total_queries
    recall_5 = totals[5] / total_queries

    print(f"Recall@1: {recall_1:.2f}")
    print(f"Recall@3: {recall_3:.2f}")
    print(f"Recall@5: {recall_5:.2f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
