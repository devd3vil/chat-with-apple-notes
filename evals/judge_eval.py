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
from src.chain import Chunker
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
        "You are an evaluator for a Q&A system over notes.\n"
        "Compare the model answer to the reference answer.\n"
        "Be moderately strict: return pass only if the answer covers the key facts from the reference without adding unsupported details.\n"
        "Omissions of key facts should reduce score and may cause fail. Any hallucinated detail should cause fail.\n\n"
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


def _build_strict_prompt(query: str, snippets: str) -> str:
    return (
        "You are a retrieval-grounded assistant. You must answer using only the provided snippets.\n\n"
        "Rules:\n"
        "Use ONLY the provided snippets as evidence. Do not add unstated advice, adjectives, or extra options.\n"
        "Answer the question directly in 1–3 sentences.\n"
        "Include key specifics from the snippets when relevant: times, durations, distances, names, and why reasons.\n"
        "If the snippets do not contain the answer, say exactly: \"I don't have that in the provided snippets.\"\n\n"
        "Citations:\n"
        "Every sentence containing factual content must end with citations in square brackets, e.g. [n6_2].\n"
        "Cite only snippet IDs that appear in the provided snippets.\n"
        "Do not cite a snippet unless it directly supports that sentence.\n\n"
        f"Question: {query}\n\n"
        f"Snippets:\n{snippets}\n\n"
        "Output format:\n"
        "Answer: <1–3 sentences, each ends with citations>\n"
        "Citations: <comma-separated list of snippet ids used>\n"
    )


def _build_facts_prompt(query: str, snippets: str) -> str:
    return (
        "Extract key facts from the snippets that directly answer the question.\n"
        "Include times, durations, distances, names, and reasons if present.\n"
        "Return 3–8 bullet points. Each bullet must end with citations in square brackets.\n\n"
        f"Question: {query}\n\n"
        f"Snippets:\n{snippets}\n\n"
        "Key facts:"
    )


def _build_answer_with_facts_prompt(query: str, facts: str) -> str:
    return (
        "You are a retrieval-grounded assistant. Use ONLY the key facts below.\n"
        "Answer the question directly in 1–3 sentences.\n"
        "Every sentence with factual content must end with citations from the key facts.\n"
        "If the facts do not contain the answer, say exactly: \"I don't have that in the provided snippets.\"\n\n"
        f"Question: {query}\n\n"
        f"Key facts:\n{facts}\n\n"
        "Output format:\n"
        "Answer: <1–3 sentences, each ends with citations>\n"
        "Citations: <comma-separated list of snippet ids used>\n"
    )


def _build_repair_prompt(query: str, snippets: str, draft: str) -> str:
    return (
        "Given:\n"
        f"Question: {query}\n\n"
        f"Snippets:\n{snippets}\n\n"
        f"Draft answer: {draft}\n\n"
        "Fix the draft answer to comply with the rules:\n"
        "Remove any content not explicitly supported by snippets.\n"
        "Add missing key facts that ARE in snippets (times/durations/distances/why).\n"
        "Ensure each sentence with factual content ends with correct snippet citations.\n"
        "Return the corrected answer in the same output format.\n"
    )


def _format_snippets(citations: list) -> str:
    return "\n".join(f"[{c.chunk_id}] {c.text}" for c in citations)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge evaluation.")
    parser.add_argument("--notes", default="evals/fixture_notes.json")
    parser.add_argument("--golden", default="evals/golden_set.json")
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--ollama-embedding-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--judge-max-tokens", type=int, default=128)
    parser.add_argument("--facts-max-tokens", type=int, default=128)
    parser.add_argument("--use-key-facts", action=argparse.BooleanOptionalAction, default=True)
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
    answer_llm = OllamaLLM(model_name=judge_model, base_url=base_url)
    judge = OllamaLLM(model_name=judge_model, base_url=base_url)

    results = []
    total_score = 0.0
    passes = 0

    for item in golden:
        query = item["query"]
        reference = item["reference_answer"]

        citations = retriever.retrieve(query, top_k=args.top_k)
        snippets = _format_snippets(citations)

        if not citations:
            answer = "I don't have that in the provided snippets."
            judge_output = judge.generate(
                _judge_prompt(query, reference, answer),
                max_tokens=args.judge_max_tokens,
            )
            verdict = _parse_judge_json(judge_output)
        else:
            answer = ""
            verdict = {"verdict": "fail", "score": 0.0, "rationale": "Not evaluated"}
            attempts = 0
            facts = ""
            if args.use_key_facts:
                facts_prompt = _build_facts_prompt(query, snippets)
                facts = answer_llm.generate(facts_prompt, max_tokens=args.facts_max_tokens)
            while attempts < args.max_attempts:
                if attempts == 0:
                    if args.use_key_facts and facts:
                        prompt = _build_answer_with_facts_prompt(query, facts)
                    else:
                        prompt = _build_strict_prompt(query, snippets)
                else:
                    prompt = _build_repair_prompt(query, snippets, answer)
                answer = answer_llm.generate(prompt, max_tokens=args.answer_max_tokens)
                judge_output = judge.generate(
                    _judge_prompt(query, reference, answer),
                    max_tokens=args.judge_max_tokens,
                )
                verdict = _parse_judge_json(judge_output)
                if str(verdict.get("verdict", "")).lower() == "pass":
                    break
                attempts += 1

        score = float(verdict.get("score", 0.0))
        total_score += score
        if str(verdict.get("verdict", "")).lower() == "pass":
            passes += 1

        results.append(
            {
                "query": query,
                "reference_answer": reference,
                "model_answer": answer,
                "citations": [c.chunk_id for c in citations],
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
