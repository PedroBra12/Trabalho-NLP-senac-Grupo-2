"""
RAG Pipeline — LangChain + ChromaDB + Ollama
============================================
Versão equivalente ao `rag_pipeline.py` (LlamaIndex), porém usando LangChain.

Models:
  Embeddings :  PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir
                BAAI/bge-m3
  Reranker   :  unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2  (CrossEncoder)
  LLM        :  gemma4:e4b  (via Ollama)

Usage:
  python rag_lang_pipeline.py                        # demo com texto de exemplo
  python rag_lang_pipeline.py --pdf path/to/doc.pdf  # indexa um PDF
  python rag_lang_pipeline.py --dir ./docs           # indexa PDFs de um diretório
  python rag_lang_pipeline.py --eval golden_qa.json  # avaliação golden Q&A
"""

import re
import gc
import json
import torch
import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from sentence_transformers import CrossEncoder


# ─── Config ───────────────────────────────────────────────────────────────────

EMBED_MODEL = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir"
RERANK_MODEL = "unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2"
OLLAMA_MODEL = "qwen3:14b"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
RETRIEVAL_K = 20  # candidatos antes do reranking
TOP_K = 5  # chunks passados ao LLM após reranking
PDF_PATH = "PDFs/SM-A15X_A16X_A17X_A06X_A075_16_Emb_BR_Rev.2.1.pdf"
GOLDEN_PATH = "Programas/RAG/golden_qa.json"
PERSIST_DIR = "./.chromadb_lang"
COLLECTION_NAME = "rag_docs_lang"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = """\
Você é um assistente prestativo. Responda à pergunta do usuário usando APENAS \
o contexto fornecido abaixo. Se a resposta não estiver no contexto, diga isso \
honestamente. Responda no mesmo idioma da pergunta.

REGRAS IMPORTANTES:
- Seja direto e conciso. Não adicione explicações além do que foi perguntado.
- Use as MESMAS palavras do contexto sempre que possível, sem reformular.
- Não use frases introdutórias como "De acordo com o contexto" ou "O documento diz".
- Responda em UMA frase quando possível.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Contexto:\n{context}\n\nPergunta: {question}"),
    ]
)


# ─── Embeddings, VectorStore, LLM, Reranker ───────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.1,
    keep_alive="30m",
    num_ctx=4096,
    num_predict=512,
    repeat_penalty=1.1,
)

reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ─── Ingestão ─────────────────────────────────────────────────────────────────


def _add_docs(docs: list[Document], source_label: str):
    chunks = splitter.split_documents(docs)
    if not chunks:
        print(f"  Nenhum chunk gerado para '{source_label}'")
        return
    vectorstore.add_documents(chunks)
    print(f"  Indexados {len(chunks)} chunks de '{source_label}'")


def index_text(text: str, doc_id: str = "manual"):
    """Indexa uma string crua."""
    doc = Document(page_content=text, metadata={"source": doc_id})
    _add_docs([doc], doc_id)


def index_pdf(path: str | Path):
    """Indexa um único PDF."""
    docs = PyMuPDFLoader(str(path)).load()
    _add_docs(docs, Path(path).name)


def index_directory(directory: str | Path):
    """Indexa todos os PDFs de um diretório (recursivo)."""
    loader = DirectoryLoader(
        str(directory),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    _add_docs(docs, str(directory))


# ─── Recuperação + Reranking ──────────────────────────────────────────────────


def retrieve_and_rerank(query: str) -> list[Document]:
    candidates = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    if not candidates:
        return []
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:TOP_K]
    # anexa o score como metadado para inspeção
    for doc, score in top:
        doc.metadata["rerank_score"] = float(score)
    return [doc for doc, _ in top]


def _format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ─── Cadeia (chain) RAG ───────────────────────────────────────────────────────


def ask(query: str, verbose: bool = False) -> str:
    print("  → recuperando chunks...", flush=True)
    docs = retrieve_and_rerank(query)
    context = _format_context(docs)

    print("  → consultando LLM...", flush=True)
    if verbose:
        print("\n── Resposta ────────────────────────────────────────────")

    prompt = PROMPT_TEMPLATE.format_messages(context=context, question=query)
    answer_parts: list[str] = []
    for chunk in llm.stream(prompt):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if verbose:
            print(token, end="", flush=True)
        answer_parts.append(token)
    answer = "".join(answer_parts).strip()

    # Retry sem streaming se a resposta veio vazia (gemma3:4b ocasionalmente
    # encerra cedo no streaming).
    if not answer:
        if verbose:
            print("[resposta vazia — tentando novamente sem streaming]", flush=True)
        result = llm.invoke(prompt)
        answer = (result.content if hasattr(result, "content") else str(result)).strip()
        if verbose:
            print(answer)

    if verbose:
        print("\n\n── Source chunks used ──────────────────────────────────")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("file_path") or d.metadata.get("source", "?")
            score = d.metadata.get("rerank_score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            preview = d.page_content[:80].replace("\n", " ")
            print(f"  [{i}] score={score_str} | {Path(str(src)).name} | {preview}...")
    return answer


# ─── Avaliação Golden Q&A ─────────────────────────────────────────────────────


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

    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    emb = model.encode([pred, ref], convert_to_tensor=True)
    sim = torch.nn.functional.cosine_similarity(
        emb[0].unsqueeze(0), emb[1].unsqueeze(0)
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
        print(f"\n  Reference: {reference}")
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
        f"  Exact Match:      {sum(all_em)}/{len(all_em)} "
        f"({100*sum(all_em)/len(all_em):.1f}%)"
    )
    print(f"  Avg Token F1:     {sum(all_f1)/len(all_f1):.4f}")
    print(f"  Avg Semantic Sim: {sum(all_sim)/len(all_sim):.4f}")
    print("=" * 60)
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline (LangChain)")
    parser.add_argument("--pdf", help="Caminho para um PDF a indexar")
    parser.add_argument("--dir", help="Diretório de PDFs a indexar")
    parser.add_argument("--eval", help="Caminho para golden_qa.json")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Limpa a coleção antes de indexar (evita duplicatas)",
    )
    args = parser.parse_args()

    # ── Indexação ─────────────────────────────────────────────────────────────
    existing_count = vectorstore._collection.count()
    if args.reindex and existing_count > 0:
        print(f"Limpando coleção existente ({existing_count} documentos)...")
        vectorstore.reset_collection()
        existing_count = 0

    skip_indexing = existing_count > 0 and not (args.pdf or args.dir)
    if skip_indexing:
        print(
            f"Coleção '{COLLECTION_NAME}' já tem {existing_count} chunks — "
            f"pulando indexação. Use --reindex para recriar."
        )
    elif args.pdf:
        print(f"Indexando PDF: {args.pdf}")
        index_pdf(args.pdf)
    elif args.dir:
        print(f"Indexando diretório: {args.dir}")
        index_directory(args.dir)
    elif PDF_PATH and Path(PDF_PATH).exists():
        print(f"Indexando PDF padrão: {PDF_PATH}")
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

    # ── Avaliação ou loop interativo ──────────────────────────────────────────
    eval_path = args.eval or (
        GOLDEN_PATH if GOLDEN_PATH and Path(GOLDEN_PATH).exists() else None
    )
    if eval_path:
        print(f"\nRodando avaliação golden Q&A: {eval_path}")
        run_golden_eval(eval_path, verbose=True)
    else:
        print("\nPipeline RAG (LangChain) pronto. Digite uma pergunta (ou 'sair').\n")
        while True:
            try:
                query = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in {"sair", "quit", "exit", "q"}:
                break
            answer = ask(query, verbose=True)
            print(f"\nAssistente: {answer}\n")
