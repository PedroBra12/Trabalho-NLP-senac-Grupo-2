"""
RAG Pipeline — ChromaDB + Qwen3-Embedding-8B + Qwen3-Reranker-8B + Qwen3:14b
==============================================================================
Features:
  - Two-stage retrieval: embedding candidates → cross-encoder reranking
  - PDF ingestion (via pymupdf)
  - Golden Q&A evaluation (Exact Match, F1, semantic similarity)
  - Retrieval metrics: NDCG@K, MRR@K, Precision/Recall@K, IoU, token count
  - LLM-as-a-Judge: faithfulness, relevance, completeness

Usage:
  python rag_pipeline.py                        # demo with sample text
  python rag_pipeline.py --pdf path/to/doc.pdf  # index a PDF
  python rag_pipeline.py --dir ./docs           # index a directory of .txt files
  python rag_pipeline.py --eval golden_qa.json  # run golden Q&A evaluation
"""

import gc
import re
import math
import json
import torch
import ollama
import argparse
import chromadb
from pathlib import Path
from dataclasses import dataclass, field
from sentence_transformers import CrossEncoder, SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ─── Config ───────────────────────────────────────────────────────────────────

COLLECTION_NAME = "rag_docs"
EMBED_MODEL     = "Qwen/Qwen3-Embedding-8B"    # Stage 1 — ~8 GB VRAM
RERANK_MODEL    = "Qwen/Qwen3-Reranker-8B"     # Stage 2 — ~8 GB VRAM
OLLAMA_MODEL    = "qwen3:14b"                  # Generation — ~9 GB VRAM
SIM_MODEL       = "all-MiniLM-L6-v2"           # Lightweight model for semantic similarity scoring
CHUNK_SIZE      = 1200
CHUNK_OVERLAP   = 100
RETRIEVAL_K     = 25    # candidates from embedding search
TOP_K           = 5     # final chunks after reranking
PDF_PATH     = "PDFs/SM-A15X_A16X_A17X_A06X_A075_16_Emb_BR_Rev.2.1.pdf"   # set to None to skip
GOLDEN_PATH  = "Programas/RAG/golden_qa.json"           # set to None to skip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    query:             str
    chunks:            list[str]
    answer:            str
    retrieval_metrics: dict       = field(default_factory=dict)
    rerank_scores:     list[float] = field(default_factory=list)
    judge_scores:      dict       = field(default_factory=dict)
    golden_metrics:    dict       = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*62}",
            f"Query:  {self.query}",
            f"\nAnswer:\n{self.answer}",
            f"\n── Retrieval metrics ────────────────────────────────────────",
        ]
        for k, v in self.retrieval_metrics.items():
            lines.append(f"  {k:<26} {v}")
        if self.rerank_scores:
            lines.append(f"\n── Reranker scores (top-{TOP_K}) ─────────────────────────────")
            for i, s in enumerate(self.rerank_scores, 1):
                lines.append(f"  [{i}] {s:.4f}")
        if self.judge_scores:
            lines.append(f"\n── LLM-as-a-Judge ───────────────────────────────────────────")
            for k, v in self.judge_scores.items():
                lines.append(f"  {k:<26} {v}")
        if self.golden_metrics:
            lines.append(f"\n── Golden Q&A evaluation ────────────────────────────────────")
            for k, v in self.golden_metrics.items():
                lines.append(f"  {k:<26} {v}")
        lines.append("="*62)
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


# ─── Ingestion ────────────────────────────────────────────────────────────────

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

def index_pdf(path: str | Path) -> int:
    """Extract text from a PDF and index it page-aware."""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("pymupdf is required for PDF support: pip install pymupdf")
    p    = Path(path)
    doc  = fitz.open(str(p))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()
    full_text = "\n\n".join(pages)
    return index_text(full_text, doc_id=p.name)

def index_directory(directory: str | Path, glob: str = "**/*.txt") -> int:
    return sum(index_file(p) for p in Path(directory).glob(glob))


# ─── Stage 1: Embedding retrieval ─────────────────────────────────────────────

def retrieve_candidates(query: str, k: int = RETRIEVAL_K) -> tuple[list[str], list[float]]:
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    results  = collection.query(query_texts=[prefixed], n_results=k)
    return results["documents"][0], results["distances"][0]


# ─── Stage 2: Reranking ───────────────────────────────────────────────────────

def rerank(query: str, candidates: list[str], top_k: int = TOP_K) -> tuple[list[str], list[float]]:
    """Cross-encoder reranker: load → score → unload → return top_k."""
    print(f"  Loading reranker ({RERANK_MODEL})...")
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, trust_remote_code=True)
    pairs    = [[query, chunk] for chunk in candidates]
    scores   = reranker.predict(pairs).tolist()
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ranked     = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_scores = [s for s, _ in ranked[:top_k]]
    top_chunks = [c for _, c in ranked[:top_k]]
    return top_chunks, top_scores


# ─── Retrieval metrics ────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(text.split())

def _iou(chunk: str, query: str) -> float:
    c, q = set(chunk.lower().split()), set(query.lower().split())
    if not c or not q:
        return 0.0
    return round(len(c & q) / len(c | q), 4)

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
        f"NDCG@{k} (pre-rerank)":    _ndcg_at_k(relevances, k),
        f"MRR@{k} (pre-rerank)":     _mrr_at_k(relevances, k),
        f"Precision@{k}":            precision,
        f"Recall@{k}":               recall,
        "avg_tokens":                round(sum(_token_count(c) for c in chunks) / len(chunks), 1),
        "avg_IoU":                   round(sum(_iou(c, query) for c in chunks) / len(chunks), 4),
        "candidates_retrieved":      len(distances),
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


# ─── Golden Q&A evaluation ────────────────────────────────────────────────────
#
# Golden Q&A format (golden_qa.json):
# [
#   {"question": "O que é ChromaDB?", "answer": "Um banco de dados vetorial open-source."},
#   ...
# ]
#
# Metrics computed:
#   Exact Match     — 1 if predicted == reference (case-insensitive, stripped)
#   Token F1        — harmonic mean of token-level precision and recall
#   Semantic sim.   — cosine similarity of sentence embeddings (all-MiniLM-L6-v2)

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)

def _token_f1(pred: str, ref: str) -> float:
    pred_tokens = _normalize(pred).split()
    ref_tokens  = _normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common    = set(pred_tokens) & set(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)

def _semantic_similarity(pred: str, ref: str) -> float:
    """Cosine similarity using a lightweight local model (no VRAM needed)."""
    model      = SentenceTransformer(SIM_MODEL, device="cpu")
    embeddings = model.encode([pred, ref], convert_to_tensor=True)
    cos_sim    = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    ).item()
    del model
    gc.collect()
    return round(float(cos_sim), 4)

def evaluate_golden(pred: str, ref: str) -> dict:
    return {
        "exact_match":      int(_normalize(pred) == _normalize(ref)),
        "token_f1":         _token_f1(pred, ref),
        "semantic_sim":     _semantic_similarity(pred, ref),
    }

def run_golden_eval(golden_path: str | Path, verbose: bool = True) -> list[dict]:
    """
    Load a golden Q&A JSON file, run each question through the pipeline,
    and compute aggregate metrics.

    golden_qa.json schema:
      [{"question": "...", "answer": "..."}, ...]
    """
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    results, all_f1, all_sim, all_em = [], [], [], []

    for item in golden:
        query     = item["question"]
        reference = item["answer"]
        print(f"\n[Golden] Q: {query}")
        result = ask(query, run_judge=False, verbose=verbose)
        metrics = evaluate_golden(result.answer, reference)
        result.golden_metrics = metrics
        all_em.append(metrics["exact_match"])
        all_f1.append(metrics["token_f1"])
        all_sim.append(metrics["semantic_sim"])
        print(result)
        results.append({
            "question":   query,
            "reference":  reference,
            "prediction": result.answer,
            "metrics":    metrics,
        })

    print(f"\n{'='*62}")
    print(f"Golden Q&A aggregate ({len(golden)} questions):")
    print(f"  Exact Match:      {sum(all_em)}/{len(all_em)} ({100*sum(all_em)/len(all_em):.1f}%)")
    print(f"  Avg Token F1:     {sum(all_f1)/len(all_f1):.4f}")
    print(f"  Avg Semantic Sim: {sum(all_sim)/len(all_sim):.4f}")
    print("="*62)
    return results


# ─── Full pipeline ────────────────────────────────────────────────────────────

def ask(query: str, run_judge: bool = True, verbose: bool = False) -> RAGResult:
    """
    Full RAG pipeline:
      1. Retrieve RETRIEVAL_K candidates   (embedder in VRAM)
      2. Compute retrieval metrics
      3. Unload embedder → rerank → keep TOP_K
      4. Generate answer                   (LLM in VRAM, keep_alive=0)
      5. LLM-as-a-Judge
      6. Reload embedder for next query
    """
    if verbose:
        print(f"\n[1/5] Retrieving {RETRIEVAL_K} candidates...")
    candidates, distances = retrieve_candidates(query)

    metrics = compute_retrieval_metrics(query, candidates, distances, k=TOP_K)
    if verbose:
        print(f"[2/5] Retrieval metrics: {metrics}")

    if verbose:
        print(f"[3/5] Reranking {len(candidates)} candidates → top {TOP_K}...")
    unload_embedder()
    chunks, rerank_scores = rerank(query, candidates)
    if verbose:
        print(f"  Reranker scores: {[round(s, 4) for s in rerank_scores]}")

    if verbose:
        print("[4/5] Generating answer...")
    answer = generate(query, chunks)

    judge_scores = {}
    if run_judge:
        if verbose:
            print("[5/5] Running LLM-as-a-Judge...")
        judge_scores = llm_judge(query, chunks, answer)

    reload_embedder()

    return RAGResult(
        query=query,
        chunks=chunks,
        answer=answer,
        retrieval_metrics=metrics,
        rerank_scores=rerank_scores,
        judge_scores=judge_scores,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--pdf",  help="Path to a PDF file to index")
    parser.add_argument("--dir",  help="Directory of .txt files to index")
    parser.add_argument("--eval", help="Path to golden_qa.json for batch evaluation")
    args = parser.parse_args()

    # ── Index sources ─────────────────────────────────────────────────────────
    if args.pdf:
        print(f"Indexing PDF: {args.pdf}")
        index_pdf(args.pdf)
    elif args.dir:
        print(f"Indexing directory: {args.dir}")
        index_directory(args.dir)
    else:
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

    # ── Golden evaluation mode ────────────────────────────────────────────────
    if args.eval:
        print(f"\nRunning golden Q&A evaluation from: {args.eval}")
        run_golden_eval(args.eval, verbose=True)
    else:
        # ── Interactive Q&A loop ──────────────────────────────────────────────
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
