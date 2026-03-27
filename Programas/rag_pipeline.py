"""
RAG Pipeline — ChromaDB + Qwen3-Embedding-8B + Qwen3-Reranker-8B + Qwen3:14b
==============================================================================
Two-stage retrieval:
  Stage 1 — Embedding model retrieves RETRIEVAL_K (~25) candidates (fast, approximate)
  Stage 2 — Cross-encoder reranker scores all candidates, keeps TOP_K (slow, precise)

Install dependencies:
    pip install chromadb sentence-transformers ollama torch

Make sure Ollama is running:
    ollama serve
    ollama pull qwen3:14b
"""

import gc
import math
import json
import torch
import ollama
import chromadb
from pathlib import Path
from dataclasses import dataclass, field
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ─── Config ───────────────────────────────────────────────────────────────────

COLLECTION_NAME = "rag_docs"
EMBED_MODEL     = "Qwen/Qwen3-Embedding-8B"    # Stage 1 — ~8 GB VRAM
RERANK_MODEL    = "Qwen/Qwen3-Reranker-8B"     # Stage 2 — ~8 GB VRAM (loaded separately)
OLLAMA_MODEL    = "qwen3:14b"                  # Generation — ~9 GB VRAM
CHUNK_SIZE      = 1200
CHUNK_OVERLAP   = 100
RETRIEVAL_K     = 25    # candidates fetched by the embedding model
TOP_K           = 5     # final chunks passed to the LLM after reranking

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    query:             str
    chunks:            list[str]
    answer:            str
    retrieval_metrics: dict = field(default_factory=dict)
    rerank_scores:     list[float] = field(default_factory=list)
    judge_scores:      dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Query:  {self.query}",
            f"\nAnswer:\n{self.answer}",
            f"\n── Retrieval metrics ──────────────────────────────────",
        ]
        for k, v in self.retrieval_metrics.items():
            lines.append(f"  {k:<22} {v}")
        if self.rerank_scores:
            lines.append(f"\n── Reranker scores (top-{TOP_K}) ───────────────────────")
            for i, s in enumerate(self.rerank_scores, 1):
                lines.append(f"  [{i}] {s:.4f}")
        if self.judge_scores:
            lines.append(f"\n── LLM-as-a-Judge ─────────────────────────────────────")
            for k, v in self.judge_scores.items():
                lines.append(f"  {k:<22} {v}")
        lines.append("="*60)
        return "\n".join(lines)


# ─── ChromaDB + embedding setup ───────────────────────────────────────────────

def _make_embed_fn():
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device=DEVICE,
    )

embed_fn   = _make_embed_fn()
chroma     = chromadb.PersistentClient(path="./.chromadb")
collection = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn,
)


# ─── VRAM management ──────────────────────────────────────────────────────────

def unload_embedder():
    global embed_fn, collection
    del embed_fn
    del collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def reload_embedder():
    global embed_fn, collection
    embed_fn   = _make_embed_fn()
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


# ─── Indexing ─────────────────────────────────────────────────────────────────

def index_text(text: str, doc_id: str) -> int:
    chunks    = chunk_text(text)
    ids       = [f"{doc_id}::{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"  Indexed {len(chunks)} chunks from '{doc_id}'")
    return len(chunks)

def index_file(path: str | Path) -> int:
    p = Path(path)
    return index_text(p.read_text(encoding="utf-8"), doc_id=p.name)

def index_directory(directory: str | Path, glob: str = "**/*.txt") -> int:
    return sum(index_file(p) for p in Path(directory).glob(glob))


# ─── Stage 1: Embedding retrieval ─────────────────────────────────────────────

def retrieve_candidates(query: str, k: int = RETRIEVAL_K) -> tuple[list[str], list[float]]:
    """Fast approximate retrieval — returns broad candidate pool."""
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    results  = collection.query(query_texts=[prefixed], n_results=k)
    return results["documents"][0], results["distances"][0]


# ─── Stage 2: Reranking ───────────────────────────────────────────────────────

def rerank(query: str, candidates: list[str], top_k: int = TOP_K) -> tuple[list[str], list[float]]:
    """
    Load the cross-encoder reranker, score all candidates jointly with the query,
    return the top_k chunks sorted by reranker score (descending).
    The reranker is unloaded from VRAM immediately after scoring.
    """
    print(f"  Loading reranker ({RERANK_MODEL})...")
    reranker = CrossEncoder(
        RERANK_MODEL,
        device=DEVICE,
        trust_remote_code=True,   # required for Qwen3-Reranker
    )

    pairs  = [[query, chunk] for chunk in candidates]
    scores = reranker.predict(pairs).tolist()

    # Unload reranker from VRAM
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Sort by score descending, keep top_k
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_scores = [s for s, _ in ranked[:top_k]]
    top_chunks = [c for _, c in ranked[:top_k]]
    return top_chunks, top_scores


# ─── Retrieval metrics ────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(text.split())

def _iou(chunk: str, query: str) -> float:
    c_tokens = set(chunk.lower().split())
    q_tokens = set(query.lower().split())
    if not c_tokens or not q_tokens:
        return 0.0
    return round(len(c_tokens & q_tokens) / len(c_tokens | q_tokens), 4)

def _relevance_scores(distances: list[float]) -> list[float]:
    max_d = max(distances) if max(distances) > 0 else 1
    return [round(1 - (d / max_d), 4) for d in distances]

def _ndcg_at_k(relevances: list[float], k: int) -> float:
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(relevances, reverse=True)[:k]))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0

def _mrr_at_k(relevances: list[float], k: int, threshold: float = 0.5) -> float:
    for i, r in enumerate(relevances[:k]):
        if r >= threshold:
            return round(1 / (i + 1), 4)
    return 0.0

def _precision_recall_at_k(relevances: list[float], k: int, threshold: float = 0.5):
    hits      = sum(1 for r in relevances[:k] if r >= threshold)
    total_rel = sum(1 for r in relevances if r >= threshold)
    return round(hits / k, 4), round(hits / total_rel, 4) if total_rel > 0 else 0.0

def compute_retrieval_metrics(query: str, chunks: list[str], distances: list[float], k: int) -> dict:
    relevances        = _relevance_scores(distances)
    precision, recall = _precision_recall_at_k(relevances, k)
    return {
        f"NDCG@{k} (pre-rerank)":  _ndcg_at_k(relevances, k),
        f"MRR@{k} (pre-rerank)":   _mrr_at_k(relevances, k),
        f"Precision@{k}":          precision,
        f"Recall@{k}":             recall,
        "avg_tokens":              round(sum(_token_count(c) for c in chunks) / len(chunks), 1),
        "avg_IoU":                 round(sum(_iou(c, query) for c in chunks) / len(chunks), 4),
        "candidates_retrieved":    len(distances),
    }


# ─── Generation ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the context
provided below. If the answer is not in the context, say so honestly.
Answer in the same language the user asked in.
"""

def build_prompt(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"

def generate(query: str, chunks: list[str]) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        keep_alive=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, chunks)},
        ],
    )
    return response["message"]["content"]


# ─── LLM-as-a-Judge ───────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an impartial RAG evaluator. Given a question, retrieved context, and an
answer, score the answer on three dimensions from 0.0 to 1.0:

- faithfulness:  Is the answer fully supported by the context? (no hallucination)
- relevance:     Does the answer address the question directly?
- completeness:  Does the answer cover all key information in the context?

Respond ONLY with valid JSON, no explanation, no markdown. Example:
{"faithfulness": 0.9, "relevance": 0.8, "completeness": 0.7, "reasoning": "..."}
"""

def llm_judge(query: str, chunks: list[str], answer: str) -> dict:
    context  = "\n\n---\n\n".join(chunks)
    prompt   = (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Now provide your JSON evaluation."
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        keep_alive=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = response["message"]["content"]
    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Judge returned invalid JSON", "raw": raw}


# ─── Full pipeline ────────────────────────────────────────────────────────────

def ask(query: str, run_judge: bool = True, verbose: bool = False) -> RAGResult:
    """
    Full RAG pipeline with two-stage retrieval:

      1. Embed retrieval   — fetch RETRIEVAL_K candidates   (embedder in VRAM)
      2. Retrieval metrics — NDCG@K, MRR@K, Prec/Rec@K, tokens, IoU
      3. Rerank            — cross-encoder scores candidates, keeps TOP_K
                             (embedder unloaded, reranker loaded then unloaded)
      4. Generate          — LLM answers using reranked top chunks
      5. LLM-as-a-Judge    — faithfulness, relevance, completeness
      6. Reload embedder   — ready for next query
    """

    # ── Step 1: Candidate retrieval ───────────────────────────────────────────
    if verbose:
        print(f"\n[1/5] Retrieving {RETRIEVAL_K} candidates...")
    candidates, distances = retrieve_candidates(query)

    # ── Step 2: Retrieval metrics (on full candidate pool) ────────────────────
    metrics = compute_retrieval_metrics(query, candidates, distances, k=TOP_K)
    if verbose:
        print(f"[2/5] Retrieval metrics: {metrics}")

    # ── Step 3: Rerank candidates → keep TOP_K ────────────────────────────────
    if verbose:
        print(f"[3/5] Reranking {len(candidates)} candidates → top {TOP_K}...")
    unload_embedder()
    chunks, rerank_scores = rerank(query, candidates)
    if verbose:
        print(f"  Reranker scores: {[round(s, 4) for s in rerank_scores]}")

    # ── Step 4: Generate answer ───────────────────────────────────────────────
    if verbose:
        print("[4/5] Generating answer...")
    answer = generate(query, chunks)

    # ── Step 5: LLM-as-a-Judge ────────────────────────────────────────────────
    judge_scores = {}
    if run_judge:
        if verbose:
            print("[5/5] Running LLM-as-a-Judge...")
        judge_scores = llm_judge(query, chunks, answer)

    # ── Reload embedder for next query ────────────────────────────────────────
    reload_embedder()

    return RAGResult(
        query=query,
        chunks=chunks,
        answer=answer,
        retrieval_metrics=metrics,
        rerank_scores=rerank_scores,
        judge_scores=judge_scores,
    )


# ─── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    SAMPLE = """
    ChromaDB é um banco de dados vetorial open-source projetado para aplicações de IA.
    Ele armazena embeddings junto com metadados e suporta busca por similaridade eficiente.
    ChromaDB pode rodar em memória ou persistir dados em disco.

    O modelo Qwen3-Embedding-8B lidera o ranking MTEB multilíngue e tem excelente suporte
    para português brasileiro. O Qwen3-Reranker-8B é seu par de reranking — ao usá-los
    juntos em um pipeline de dois estágios, obtém-se qualidade de recuperação superior.

    Ollama permite executar grandes modelos de linguagem localmente. Suporta modelos como
    Qwen3, Llama 3, Mistral, Gemma e Phi. Execute `ollama serve` para iniciar o servidor
    e `ollama pull <modelo>` para baixar um modelo.

    RAG (Retrieval-Augmented Generation) combina recuperação e geração. O pipeline de dois
    estágios primeiro recupera candidatos com embeddings (rápido), depois reordena com um
    cross-encoder (preciso), e finalmente gera a resposta com um LLM.
    """

    print("Indexando dados de exemplo...")
    index_text(SAMPLE, doc_id="sample_ptbr")

    if len(sys.argv) > 1:
        print(f"\nIndexando arquivos de: {sys.argv[1]}")
        index_directory(sys.argv[1])

    print("\nPipeline RAG pronto. Digite uma pergunta (ou 'sair' para encerrar).\n")
    while True:
        try:
            query = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"sair", "quit", "exit", "q"}:
            break
        result = ask(query, run_judge=True, verbose=True)
        print(result)
