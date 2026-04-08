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

EMBED_MODEL = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir"
RERANK_MODEL = "unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2"
OLLAMA_MODEL = "gemma4:e4b"
CHUNK_SIZE = 1536
CHUNK_OVERLAP = 300
RETRIEVAL_K = 30  # candidates before reranking
TOP_K = 8  # chunks passed to LLM after reranking
PDF_PATH = None  # set to None to skip
GOLDEN_PATH = None  # set to None to skip eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = """\
Você é um assistente prestativo e detalhado. Responda à pergunta do usuário usando APENAS \
o contexto fornecido abaixo. Forneça respostas completas e abrangentes, incluindo todos os \
detalhes relevantes do contexto. Se houver passos ou instruções, liste-os claramente. \
Se a resposta não estiver no contexto, diga isso honestamente. Responda no mesmo idioma da pergunta.
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
        "num_predict": 512,  # Max tokens to generate (increase for longer answers)
        "top_k": 40,
        "top_p": 0.9,
    },
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

    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
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
    import shutil
    import gradio as gr
    from pathlib import Path

    DOCS_DIR = Path("./docs")
    DOCS_DIR.mkdir(exist_ok=True)

    def chat(message: str, history: list) -> str:
        if not message.strip():
            return ""
        return ask(message, verbose=False)

    def upload_pdfs(files) -> str:
        if not files:
            return "Nenhum arquivo enviado."
        indexed = []
        for f in files:
            dest = DOCS_DIR / Path(f.name).name
            shutil.copy(f.name, dest)
            index_pdf(dest)
            indexed.append(dest.name)
        return f"✅ Indexado(s): {', '.join(indexed)}"

    def run_eval() -> str:
        if not GOLDEN_PATH or not Path(GOLDEN_PATH).exists():
            return "❌ golden_qa.json não encontrado."
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
                    return [], []

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
                clear.click(clear_chat, None, [chatbot])
                retry.click(retry_last, chatbot, chatbot)
                undo.click(undo_last, chatbot, chatbot)

            with gr.Tab("📄 Indexar PDFs"):
                pdf_upload = gr.File(
                    label="Selecione PDFs", file_types=[".pdf"], file_count="multiple"
                )
                upload_btn = gr.Button("📥 Indexar", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)
                upload_btn.click(
                    fn=upload_pdfs, inputs=pdf_upload, outputs=upload_status
                )
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
