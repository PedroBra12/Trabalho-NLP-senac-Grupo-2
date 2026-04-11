"""
RAG Pipeline — LlamaIndex + ChromaDB + Ollama + Gradio
======================================================
Models:
  Embeddings :  BAAI/bge-m3            (~2 GB VRAM, via HuggingFace)
                PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir (~416 MB, via HuggingFace)
                PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir (~1276 MB, via HuggingFace)
  Reranker   :  ms-marco-MiniLM-L-6-v2 (~90 MB, via HuggingFace)
                unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2 (~408 MB, via HuggingFace)
                unicamp-dl/mt5-base-en-pt-msmarco-v2 (~1489 MB, via HuggingFace)
  LLM        :  qwen3:14b                 (~9 GB VRAM, via Ollama)
                gemma4:e4b                (~3.6 GB VRAM, via Ollama)
                gemma4:26b                (~16 GB VRAM, via Ollama)

Usage:
    1. Run `python rag_pipeline.py` to start the Gradio interface.
"""

import re
import json
import torch
import argparse
import chromadb
import shutil
import gradio as gr
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
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from sentence_transformers import SentenceTransformer
import subprocess


# ─── Config ───────────────────────────────────────────────────────────────────


EMBED_MODEL = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir"
RERANK_MODEL = "unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2"
OLLAMA_MODEL = "gemma4:e4b"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
RETRIEVAL_K = 50  # candidates before reranking
TOP_K = 10  # chunks passed to LLM after reranking
PDF_PATH = None  # set to None to skip
GOLDEN_PATH = Path(__file__).parent / "golden_qa.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = """\
Você é um assistente especializado no manual do dispositivo Samsung. \
Responda à pergunta usando APENAS o contexto fornecido. \
O contexto contém trechos do manual — a resposta ESTÁ lá, encontre-a. \
Liste todos os itens relevantes encontrados. \
Seja direto e completo. Responda em português.\
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
    temperature=0.3,  # Slightly higher for more creativity (0.0-1.0)
    context_window=8192,  # Larger context window if your model supports it
    additional_kwargs={
        "num_predict": 2000,  # Max tokens to generate (increase for longer answers)
        "top_k": 40,
        "top_p": 0.9,
    },
)
Settings.text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)


# ─── ChromaDB + VectorStore ───────────────────────────────────────────────────


chroma_client = chromadb.PersistentClient(path="./.chromadb")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reset-db", action="store_true", help="Limpa o ChromaDB antes de iniciar"
)
args = parser.parse_args()

if args.reset_db:
    try:
        chroma_client.delete_collection("rag_docs")
        print("🗑️ ChromaDB resetado.")
    except Exception:
        pass  # collection didn't exist yet

chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load existing index or create empty one
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)


def reset_db() -> str:
    global _query_engine
    try:
        chroma_client.delete_collection("rag_docs")
        chroma_client.get_or_create_collection("rag_docs")
        _query_engine = None
        return "🗑️ ChromaDB resetado com sucesso."
    except Exception as e:
        return f"❌ Erro: {e}"


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
    _query_engine = None  # invalidate cache
    print(f"  Indexed text from '{doc_id}'")


def _already_indexed(filename: str) -> bool:
    results = chroma_collection.get(where={"file_name": filename}, limit=1)
    return len(results["ids"]) > 0


def index_pdf(path: str | Path):
    global _query_engine
    name = Path(path).name
    if _already_indexed(name):
        print(f"  Skipping '{name}' — already indexed")
        return
    loader = PyMuPDFReader()
    docs = loader.load(file_path=str(path))
    # tag source for dedup
    for doc in docs:
        doc.metadata["file_name"] = name
    nodes = _splitter.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)
    _query_engine = None
    print(f"  Indexed {len(nodes)} nodes from '{name}'")


_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def index_directory(directory: str | Path):
    global _query_engine
    docs = SimpleDirectoryReader(
        input_dir=str(directory),
        required_exts=[".pdf"],
        recursive=True,
    ).load_data()
    nodes = _splitter.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)
    _query_engine = None
    print(f"  Indexed {len(nodes)} nodes from '{directory}'")


# ─── Query engine ─────────────────────────────────────────────────────────────

_query_engine = None


def get_query_engine():
    global _query_engine
    if _query_engine is None:
        base_engine = index.as_query_engine(
            similarity_top_k=RETRIEVAL_K,
            node_postprocessors=[reranker],
            response_mode="compact",
        )
        hyde = HyDEQueryTransform(include_original=True)
        _query_engine = TransformQueryEngine(base_engine, hyde)
    return _query_engine


# ─── Ask ──────────────────────────────────────────────────────────────────────


def ask(query: str, verbose: bool = False) -> str:
    engine = get_query_engine()
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


_eval_model = None


def _get_eval_model():
    global _eval_model
    if _eval_model is None:
        _eval_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    return _eval_model


def _semantic_similarity(pred: str, ref: str) -> float:
    model = _get_eval_model()
    embeddings = model.encode([pred, ref], convert_to_tensor=True)
    sim = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    ).item()
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
    DOCS_DIR = Path("./docs")
    DOCS_DIR.mkdir(exist_ok=True)

    def chat(message: str, history: list) -> str:
        if not message.strip():
            return ""
        return ask(message, verbose=False)

    def upload_files(pdfs, json_file) -> str:
        msgs = []

        # Handle PDFs
        if pdfs:
            indexed, skipped = [], []
            for f in pdfs:
                dest = DOCS_DIR / Path(f.name).name
                shutil.copy(f.name, dest)
                before = _already_indexed(dest.name)
                index_pdf(dest)
                (skipped if before else indexed).append(dest.name)
            if indexed:
                msgs.append(f"✅ PDFs indexados: {', '.join(indexed)}")
            if skipped:
                msgs.append(f"⚠️ Já existiam: {', '.join(skipped)}")

        # Handle golden JSON
        if json_file:
            dest = Path(__file__).parent / "golden_qa.json"
            shutil.copy(json_file.name, dest)
            try:
                data = json.loads(dest.read_text(encoding="utf-8"))
                msgs.append(f"✅ Golden Q&A carregado: {len(data)} pares")
            except Exception as e:
                msgs.append(f"❌ JSON inválido: {e}")

        return "\n".join(msgs) if msgs else "Nenhum arquivo enviado."

    def run_eval() -> str:
        if not Path(GOLDEN_PATH).exists():
            return "❌ Nenhum golden_qa.json encontrado. Envie um na aba de indexação."
        results = run_golden_eval(GOLDEN_PATH, verbose=False)
        em = sum(r["metrics"]["exact_match"] for r in results)
        f1 = sum(r["metrics"]["token_f1"] for r in results) / len(results)
        sim = sum(r["metrics"]["semantic_sim"] for r in results) / len(results)
        lines = [
            f"**Exact Match:** {em}/{len(results)} ({100*em/len(results):.1f}%)",
            f"**Token F1:** {f1:.4f}",
            f"**Semantic Similarity:** {sim:.4f}",
            "",
            "---",
        ]
        for r in results:
            lines += [
                f"**Q:** {r['question']}",
                f"**Ref:** {r['reference']}",
                f"**Pred:** {r['prediction'][:200]}{'...' if len(r['prediction']) > 200 else ''}",
                f"EM={r['metrics']['exact_match']} | F1={r['metrics']['token_f1']:.3f} | Sim={r['metrics']['semantic_sim']:.3f}",
                "",
            ]
        return "\n".join(lines)

    with gr.Blocks(title="Samsung Manual Assistant") as demo:
        gr.HTML(
            "<h1 style='text-align:center;padding:1rem 0 0.25rem'>📱 Samsung Manual Assistant</h1>"
        )
        gr.HTML(
            "<p style='text-align:center;color:#64748b;margin-bottom:1.5rem'>Faça perguntas sobre o manual do seu dispositivo Samsung.</p>"
        )
        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                # Gradio 6.0 expects messages as dictionaries with 'role' and 'content'
                chatbot = gr.Chatbot(
                    height=480,
                    placeholder="Olá! Faça uma pergunta sobre o manual.",
                )
                msg = gr.Textbox(placeholder="Ex: O que o Painel Edge faz?", scale=7)

                with gr.Row():
                    clear = gr.Button("🗑️ Limpar", size="sm")
                    retry = gr.Button("🔄 Tentar novamente", size="sm")
                    undo = gr.Button("↩️ Desfazer", size="sm")

                # Handle chat submission with new format
                def respond(message, chat_history):
                    if not message.strip():
                        return "", chat_history

                    # chat_history is now a list of dicts: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                    # Convert to the format expected by your chat function (list of tuples)
                    tuple_history = []
                    for i in range(0, len(chat_history), 2):
                        if i + 1 < len(chat_history):
                            user_msg = chat_history[i]["content"]
                            assistant_msg = chat_history[i + 1]["content"]
                            tuple_history.append((user_msg, assistant_msg))

                    bot_message = chat(message, tuple_history)

                    # Append in the new dict format
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": bot_message})

                    return "", chat_history

                def clear_chat():
                    return []

                def retry_last(chat_history):
                    if chat_history and len(chat_history) >= 2:
                        # Get the last user message
                        last_user_msg = chat_history[-2]["content"]
                        # Remove last exchange (user + assistant)
                        chat_history = chat_history[:-2]

                        # Convert to tuple format for your chat function
                        tuple_history = []
                        for i in range(0, len(chat_history), 2):
                            if i + 1 < len(chat_history):
                                user_msg = chat_history[i]["content"]
                                assistant_msg = chat_history[i + 1]["content"]
                                tuple_history.append((user_msg, assistant_msg))

                        bot_message = chat(last_user_msg, tuple_history)

                        # Append new exchange
                        chat_history.append({"role": "user", "content": last_user_msg})
                        chat_history.append(
                            {"role": "assistant", "content": bot_message}
                        )

                    return chat_history

                def undo_last(chat_history):
                    if chat_history:
                        # Remove the last assistant message, or both if last is assistant
                        if chat_history[-1]["role"] == "assistant":
                            chat_history.pop()  # Remove assistant message
                            if chat_history and chat_history[-1]["role"] == "user":
                                chat_history.pop()  # Remove the preceding user message
                    return chat_history

                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                clear.click(clear_chat, None, chatbot)
                retry.click(retry_last, chatbot, chatbot)
                undo.click(undo_last, chatbot, chatbot)

            with gr.Tab("📄 Gerenciamento de arquivos"):
                pdf_upload = gr.File(
                    label="PDFs", file_types=[".pdf"], file_count="multiple"
                )
                json_upload = gr.File(
                    label="Golden Q&A (JSON)", file_types=[".json"], file_count="single"
                )
                upload_btn = gr.Button("📥 Enviar", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)
                upload_btn.click(
                    fn=upload_files,
                    inputs=[pdf_upload, json_upload],
                    outputs=upload_status,
                )
                reset_btn = gr.Button("🗑️ Resetar ChromaDB", variant="stop")
                reset_status = gr.Textbox(label="Status reset", interactive=False)
                reset_btn.click(fn=reset_db, inputs=None, outputs=reset_status)
            with gr.Tab("📊 Avaliação Golden Q&A"):
                eval_btn = gr.Button("▶️ Executar avaliação", variant="primary")
                eval_output = gr.Markdown()
                eval_btn.click(fn=run_eval, inputs=None, outputs=eval_output)

    # Launch with theme
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="blue", font=gr.themes.GoogleFont("IBM Plex Sans")
        ),
    )
