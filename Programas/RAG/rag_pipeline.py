"""
RAG Pipeline — LlamaIndex + ChromaDB + Ollama
==============================================
Models:
  Embeddings : BAAI/bge-m3            (~2 GB VRAM, via HuggingFace)
               PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir (~416 MB, via HuggingFace)
  Reranker   : ms-marco-MiniLM-L-6-v2 (~90 MB, via HuggingFace)
               unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2 (~408 MB, via HuggingFace)
  LLM        : qwen3:14b              (~9 GB VRAM, via Ollama)

Usage:
  python rag_pipeline.py                        # demo with sample text
  python rag_pipeline.py --pdf path/to/doc.pdf  # index a single PDF
  python rag_pipeline.py --dir ./docs           # index all PDFs in a directory
  python rag_pipeline.py --eval golden_qa.json  # run golden Q&A evaluation
  or set paths in the config section below and run without args
"""

import re
import gc
import json
import torch
import argparse
import chromadb
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# ─── Config ───────────────────────────────────────────────────────────────────

EMBED_MODEL = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir"  # "BAAI/bge-m3"
RERANK_MODEL = "unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2"  # "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "qwen3:14b"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
RETRIEVAL_K = 20  # candidates before reranking
TOP_K = 5  # chunks passed to LLM after reranking
PDF_PATH = (
    "PDFs/SM-A15X_A16X_A17X_A06X_A075_16_Emb_BR_Rev.2.1.pdf"  # set to None to skip
)
GOLDEN_PATH = "Programas/RAG/golden_qa.json"  # set to None to skip eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = """\
Você é um assistente prestativo. Responda à pergunta do usuário usando APENAS \
o contexto fornecido abaixo. Se a resposta não estiver no contexto, diga isso \
honestamente. Responda no mesmo idioma da pergunta.
"""


# ─── LlamaIndex global settings ───────────────────────────────────────────────

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device=DEVICE,
)
Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=600.0,
    keep_alive="10m",
    system_prompt=SYSTEM_PROMPT,
)
Settings.text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ─── ChromaDB + VectorStore ───────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path="./.chromadb")
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load existing index or create empty one
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)


# ─── Reranker ─────────────────────────────────────────────────────────────────

reranker = SentenceTransformerRerank(
    model=RERANK_MODEL,
    top_n=TOP_K,
)


# ─── Ingestion ────────────────────────────────────────────────────────────────


def index_text(text: str, doc_id: str = "manual"):
    """Index a raw string."""
    doc = Document(text=text, metadata={"source": doc_id})
    index.insert(doc)
    print(f"  Indexed text from '{doc_id}'")


def index_pdf(path: str | Path):
    """Index a single PDF file."""
    docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    for doc in docs:
        index.insert(doc)
    print(f"  Indexed {len(docs)} pages from '{Path(path).name}'")


def index_directory(directory: str | Path):
    """Index all PDFs in a directory."""
    docs = SimpleDirectoryReader(
        input_dir=str(directory),
        required_exts=[".pdf"],
        recursive=True,
    ).load_data()
    for doc in docs:
        index.insert(doc)
    print(f"  Indexed {len(docs)} pages from '{directory}'")


# ─── Query engine ─────────────────────────────────────────────────────────────


def make_query_engine():
    return index.as_query_engine(
        similarity_top_k=RETRIEVAL_K,
        node_postprocessors=[reranker],
        response_mode="compact",
    )


# ─── Ask ──────────────────────────────────────────────────────────────────────


def ask(query: str, verbose: bool = False) -> str:
    engine = make_query_engine()
    response = engine.query(query)
    if verbose:
        print(f"\n── Source chunks used ──────────────────────────────────")
        for i, node in enumerate(response.source_nodes, 1):
            src = node.metadata.get("file_name") or node.metadata.get("source", "?")
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  [{i}] score={score} | {src} | {node.text[:80]}...")
    return str(response)


# ─── Golden Q&A evaluation ────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _token_f1(pred: str, ref: str) -> float:
    pred_tokens = _normalize(pred).split()
    ref_tokens = _normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _semantic_similarity(pred: str, ref: str) -> float:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = model.encode([pred, ref], convert_to_tensor=True)
    sim = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    ).item()
    del model
    gc.collect()
    return round(float(sim), 4)


def run_golden_eval(golden_path: str | Path, verbose: bool = True) -> list[dict]:
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    results, all_f1, all_sim, all_em = [], [], [], []

    for item in golden:
        query = item["question"]
        reference = item["answer"]
        print(f"\n[Golden] Q: {query}")
        prediction = ask(query, verbose=verbose)
        em = int(_normalize(prediction) == _normalize(reference))
        f1 = _token_f1(prediction, reference)
        sim = _semantic_similarity(prediction, reference)
        all_em.append(em)
        all_f1.append(f1)
        all_sim.append(sim)
        print(f"  Answer:    {prediction[:120]}...")
        print(f"  Reference: {reference[:120]}")
        print(f"  EM={em}  F1={f1:.4f}  SemanticSim={sim:.4f}")
        results.append(
            {
                "question": query,
                "reference": reference,
                "prediction": prediction,
                "metrics": {"exact_match": em, "token_f1": f1, "semantic_sim": sim},
            }
        )

    print(f"\n{'='*60}")
    print(f"Golden Q&A aggregate ({len(golden)} questions):")
    print(
        f"  Exact Match:      {sum(all_em)}/{len(all_em)} ({100*sum(all_em)/len(all_em):.1f}%)"
    )
    print(f"  Avg Token F1:     {sum(all_f1)/len(all_f1):.4f}")
    print(f"  Avg Semantic Sim: {sum(all_sim)/len(all_sim):.4f}")
    print("=" * 60)
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline (LlamaIndex)")
    parser.add_argument("--pdf", help="Path to a PDF file to index")
    parser.add_argument("--dir", help="Directory of PDFs to index")
    parser.add_argument("--eval", help="Path to golden_qa.json for evaluation")
    args = parser.parse_args()

    # ── Index sources ─────────────────────────────────────────────────────────
    if args.pdf:
        print(f"Indexing PDF: {args.pdf}")
        index_pdf(args.pdf)
    elif args.dir:
        print(f"Indexing directory: {args.dir}")
        index_directory(args.dir)
    elif PDF_PATH and Path(PDF_PATH).exists():
        print(f"Indexing default PDF: {PDF_PATH}")
        index_pdf(PDF_PATH)
    else:
        SAMPLE = """
        ChromaDB é um banco de dados vetorial open-source projetado para aplicações de IA.
        O modelo BAAI/bge-m3 é excelente para português brasileiro e suporta até 8192 tokens.
        Ollama permite executar modelos de linguagem localmente, como o qwen3:14b.
        RAG combina recuperação de documentos com geração de texto para responder perguntas
        com base em dados privados, sem enviar informações para APIs externas.
        """
        print("Indexando dados de exemplo...")
        index_text(SAMPLE, doc_id="sample")

    # ── Evaluation or interactive loop ────────────────────────────────────────
    eval_path = args.eval or (
        GOLDEN_PATH if GOLDEN_PATH and Path(GOLDEN_PATH).exists() else None
    )
    if eval_path:
        print(f"\nRunning golden Q&A evaluation: {eval_path}")
        run_golden_eval(eval_path, verbose=True)
    else:
        print("\nPipeline RAG pronto. Digite uma pergunta (ou 'sair' para encerrar).\n")
        while True:
            try:
                query = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in {"sair", "quit", "exit", "q"}:
                break
            answer = ask(query, verbose=True)
            print(f"\nAssistente: {answer}\n")
