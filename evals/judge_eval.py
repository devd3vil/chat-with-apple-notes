"""
LLM-as-judge evaluation for RAG answers.
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

from src.bm25 import BM25Index
from src.chain import Chunker, RAGChain
from src.config import Settings
from src.embedder import FakeEmbedder, OllamaEmbedder
from src.llm import OllamaLLM
from src.models import Note
from src.retriever import HybridRetriever
from src.store import ChromaStore, InMemoryStore, VectorStore


def _load_notes(path: Path) -> list[Note]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Note.model_validate(item) for item in data]


def _load_golden(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_store(
    notes: list[Note],
    chunker: Chunker,
    embedder,
    use_chroma: bool,
) -> VectorStore:
    if use_chroma:
        temp_dir = TemporaryDirectory()
        store = ChromaStore(db_path=Path(temp_dir.name), embedder=embedder)
        store._temp_dir = temp_dir
    else:
        store = InMemoryStore()

    for note in notes:
        chunks = chunker.chunk_text(note.body, note.id)
        for chunk in chunks:
            chunk.embedding = embedder.embed(chunk.text)
        store.add_chunks(chunks)

    return store


def _judge_prompt(query: str, reference: str, answer: str) -> str:
    return (
        "You are a strict evaluator for a Q&A system over notes.\n"
        "Compare the model answer to the reference answer. If the model answer is "
        "correct and directly supported, return pass. If it is missing key facts or "
        "hallucinates, return fail.\n\n"
        "Return ONLY valid JSON with keys: verdict (pass/fail), score (0-1), rationale.\n\n"
        f"Query: {query}\n"
        f"Reference answer: {reference}\n"
        f"Model answer: {answer}\n"
    )


def _parse_judge_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"verdict": "fail", "score": 0.0, "rationale": "Invalid judge output"}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge evaluation.")
    parser.add_argument("--notes", default="evals/fixture_notes.json")
    parser.add_argument("--golden", default="evals/golden_set.json")
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--ollama-embedding-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-chroma", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    settings = Settings()
    base_url = args.ollama_base_url or settings.ollama_base_url
    embed_model = args.ollama_embedding_model or settings.ollama_embedding_model
    judge_model = args.judge_model or settings.ollama_chat_model

    notes = _load_notes(Path(args.notes))
    golden = _load_golden(Path(args.golden))

    if args.use_ollama:
        embedder = OllamaEmbedder(model_name=embed_model, base_url=base_url)
    else:
        embedder = FakeEmbedder(dimension=64)

    chunker = Chunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    store = _build_store(notes, chunker, embedder, use_chroma=args.use_chroma)

    bm25 = BM25Index(Path("data/bm25_judge_index.json"))
    for note in notes:
        chunks = chunker.chunk_text(note.body, note.id)
        bm25.add_documents(chunks)

    retriever = HybridRetriever(store=store, embedder=embedder, bm25=bm25)
    chain = RAGChain(retriever=retriever, llm=OllamaLLM(model_name=judge_model, base_url=base_url), top_k=args.top_k)

    judge = OllamaLLM(model_name=judge_model, base_url=base_url)

    results = []
    total_score = 0.0
    passes = 0

    for item in golden:
        query = item["query"]
        reference = item["reference_answer"]

        result = chain.ask(query)
        judge_prompt = _judge_prompt(query, reference, result.answer)
        judge_output = judge.generate(judge_prompt, max_tokens=256)
        verdict = _parse_judge_json(judge_output)

        score = float(verdict.get("score", 0.0))
        total_score += score
        if str(verdict.get("verdict", "")).lower() == "pass":
            passes += 1

        results.append(
            {
                "query": query,
                "reference_answer": reference,
                "model_answer": result.answer,
                "citations": [c.chunk_id for c in result.citations],
                "judge": verdict,
            }
        )

    total = len(golden) if golden else 1
    avg_score = total_score / total
    pass_rate = passes / total

    print(f"Judge pass rate: {pass_rate:.2f}")
    print(f"Judge avg score: {avg_score:.2f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
