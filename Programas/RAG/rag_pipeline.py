"""
RAG Pipeline — LlamaIndex + Qdrant + Ollama + Gradio
=====================================================
Embedding strategy: BGE-M3 HYBRID (dense + sparse via fastembed BM42)
  Dense  : BAAI/bge-m3 q8 — semantic similarity
  Sparse : Qdrant/bm42-all-minilm-l6-v2-attentions — exact keyword match
  Fusion : Reciprocal Rank Fusion (RRF) inside Qdrant at query time
  Rerank : BAAI/bge-reranker-v2-m3 — cross-encoder re-score top candidates

Models:
  Embeddings :  BAAI/bge-m3
  Sparse     :  Qdrant/bm42-all-minilm-l6-v2-attentions
  Reranker   :  BAAI/bge-reranker-v2-m3
  LLM        :  gemma4:e2b

Ingestion:
  PDF        : Docling (layout-aware, table extraction, OCR fallback)
  Images     : Docling (OCR) — supports PNG, JPG, TIFF, BMP, WEBP
  Audio/Video: Docling ASR (Whisper Turbo) — MP3, WAV, M4A, OGG, FLAC, MP4, MOV, AVI

Usage:
    python rag_pipeline.py [--reset-db]
"""

import os
import re
import json
import torch
import argparse
import shutil
import gradio as gr
import subprocess
from pathlib import Path

from docling.datamodel import asr_model_specs
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    AsrPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

from docling.document_converter import (
    PdfFormatOption,
    ImageFormatOption,
    AudioFormatOption,
)
from docling.pipeline.asr_pipeline import AsrPipeline

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.vector_stores.qdrant.utils import default_sparse_encoder
from sentence_transformers import SentenceTransformer

import fastembed

fastembed.SparseTextEmbedding.__init__.__defaults__

from llama_index.vector_stores.qdrant import utils as _qdrant_utils
from typing import List
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

os.environ["FASTEMBED_DEVICE"] = "cpu"
os.environ["FASTEMBED_CACHE_DIR"] = "/tmp/fastembed_cache"
os.environ["ONNXRUNTIME_PROVIDERS"] = "CPUExecutionProvider"
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

# ─── Config ───────────────────────────────────────────────────────────────────

EMBED_MODEL = "BAAI/bge-m3"  # PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
SPARSE_DOC_MODEL = "naver/efficient-splade-VI-BT-large-doc"
SPARSE_QUERY_MODEL = "naver/efficient-splade-VI-BT-large-query"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2
WHISPER_MODEL = "openai/whisper-turbo"  # or "openai/whisper-large-v3"
OLLAMA_MODEL = "gemma4:e2b"
EMBED_DIM = 1024  # BGE-M3 dense output dim
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
RETRIEVAL_K = 10  # dense candidates before RRF
SPARSE_K = 10  # sparse candidates before RRF
TOP_K = 5  # after reranker
QDRANT_PATH = "./.qdrant"
COLLECTION = "rag_docs"
GOLDEN_PATH = Path(__file__).parent / "golden_qa.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".avi"}

SYSTEM_PROMPT = """\
Você é um assistente especializado no manual do dispositivo Samsung. \
Responda à pergunta usando APENAS o contexto fornecido. \
REGRA CRÍTICA: Copie o texto EXATO do manual sempre que possível. NÃO parafraseie. \
Se a resposta for uma lista, copie cada item exatamente como aparece no manual. \
Se a resposta for um valor, número ou nome técnico, reproduza-o literalmente. \
Seja direto e completo. Responda em português.\
"""

TEXT_QA_TEMPLATE = PromptTemplate(
    "Contexto do manual:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Pergunta: {query_str}\n\n"
    "Instrução: Copie trechos EXATOS do contexto acima para responder. "
    "Não parafraseie. Se for lista, copie cada item literalmente.\n"
    "Resposta:"
)


# ─── LlamaIndex global settings ───────────────────────────────────────────────

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device=DEVICE,
    model_kwargs={"torch_dtype": torch.float16},
    embed_batch_size=4,
)

Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=600.0,
    keep_alive="10m",
    temperature=0.3,
    context_window=2048,
    additional_kwargs={
        "num_predict": 512,
        "top_k": 40,
        "top_p": 0.9,
        "num_gpu": 35,  # adjust based on your GPU memory
    },
)

Settings.text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)


# ─── Qdrant + VectorStore ─────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reset-db", action="store_true", help="Limpa o Qdrant antes de iniciar"
)
args = parser.parse_args()

qdrant_client = QdrantClient(path=QDRANT_PATH)

if args.reset_db:
    try:
        qdrant_client.delete_collection(COLLECTION)
        print("🗑️ Qdrant resetado.")
    except Exception:
        pass

# Create collection with named dense + sparse vectors if not exists
if not qdrant_client.collection_exists(COLLECTION):
    qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            # Named dense vector — required when mixing dense + sparse
            "text-dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            # Sparse vector — BM42 weights produced by fastembed at index/query time
            "text-sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )


def _cpu_default_sparse_encoder(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to("cpu")  # force CPU

    def compute_vectors(texts: List[str]):
        tokens = tokenizer(
            texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )
        # no .to("cuda") here
        output = model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        tvecs, _ = torch.max(weighted_log, dim=1)
        indices, vecs = [], []
        for batch in tvecs:
            indices.append(batch.nonzero(as_tuple=True)[0].tolist())
            vecs.append(batch[indices[-1]].tolist())
        return indices, vecs

    return compute_vectors


_qdrant_utils.default_sparse_encoder = _cpu_default_sparse_encoder


def make_cpu_sparse_encoder(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to("cpu")
    model.eval()

    def encode(texts):
        with torch.no_grad():
            tokens = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )  # stays on CPU
            output = model(**tokens)
            logits, mask = output.logits, tokens.attention_mask
            relu_log = torch.log(1 + torch.relu(logits))
            weighted = relu_log * mask.unsqueeze(-1)
            tvecs, _ = torch.max(weighted, dim=1)
            indices, vecs = [], []
            for batch in tvecs:
                idx = batch.nonzero(as_tuple=True)[0].tolist()
                indices.append(idx)
                vecs.append(batch[idx].tolist())
        return indices, vecs

    return encode


# Then pass directly to QdrantVectorStore
_sparse_encoder = make_cpu_sparse_encoder(SPARSE_DOC_MODEL)
_sparse_query_encoder = make_cpu_sparse_encoder(SPARSE_QUERY_MODEL)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION,
    dense_vector_name="text-dense",
    sparse_vector_name="text-sparse",
    enable_hybrid=True,
    sparse_doc_fn=_sparse_encoder,  # ← bypass library entirely
    sparse_query_fn=_sparse_query_encoder,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)


def reset_db() -> str:
    global _query_engine
    try:
        qdrant_client.delete_collection(COLLECTION)
        qdrant_client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "text-dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                ),
            },
        )
        _query_engine = None
        return "🗑️ Qdrant resetado com sucesso."
    except Exception as e:
        return f"❌ Erro: {e}"


# ─── Reranker ─────────────────────────────────────────────────────────────────

reranker = SentenceTransformerRerank(
    model=RERANK_MODEL,
    top_n=TOP_K,
    device=DEVICE,
)


# ─── Docling converters ───────────────────────────────────────────────────────

table_opts = TableStructureOptions(do_cell_matching=True, mode=TableFormerMode.FAST)

accel_opts = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)

_pdf_options = PdfPipelineOptions()
_pdf_options.do_ocr = True
_pdf_options.ocr_options = RapidOcrOptions(force_full_page_ocr=False)
_pdf_options.do_table_structure = True
_pdf_options.table_structure_options = table_opts
_pdf_options.accelerator_options = accel_opts

_WHISPER_MAP = {
    "openai/whisper-turbo": asr_model_specs.WHISPER_TURBO,
    "openai/whisper-large-v3": asr_model_specs.WHISPER_LARGE,
}

_asr_options = AsrPipelineOptions()
_asr_options.asr_options = _WHISPER_MAP[WHISPER_MODEL]

_doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options),
        InputFormat.IMAGE: ImageFormatOption(),
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_cls=AsrPipeline,
            pipeline_options=_asr_options,
        ),
    }
)


# ─── Splitter ─────────────────────────────────────────────────────────────────

_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ─── Ingestion helpers ────────────────────────────────────────────────────────


def _already_indexed(filename: str) -> bool:
    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="file_name", match=MatchValue(value=filename))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(points) > 0


def index_text(text: str, doc_id: str = "manual"):
    global _query_engine
    doc = Document(text=text, metadata={"source": doc_id})
    index.insert(doc)
    _query_engine = None
    print(f"  Indexed text from '{doc_id}'")


def _docling_to_docs(path: Path, source_type: str) -> list[Document]:
    result = _doc_converter.convert(str(path))
    md_text = result.document.export_to_markdown()
    return [
        Document(
            text=md_text,
            metadata={
                "file_name": path.name,
                "source_type": source_type,
                "source": str(path),
            },
        )
    ]


def index_pdf(path: str | Path):
    global _query_engine
    path = Path(path)
    name = path.name
    if _already_indexed(name):
        print(f"  Skipping '{name}' — already indexed")
        return
    print(f"  Converting PDF with Docling: '{name}'")
    docs = _docling_to_docs(path, source_type="pdf")
    nodes = _splitter.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)
    _query_engine = None
    print(f"  Indexed {len(nodes)} nodes from '{name}'")


def index_image(path: str | Path):
    global _query_engine
    path = Path(path)
    name = path.name
    if _already_indexed(name):
        print(f"  Skipping '{name}' — already indexed")
        return
    print(f"  Running OCR with Docling: '{name}'")
    docs = _docling_to_docs(path, source_type="image")
    nodes = _splitter.get_nodes_from_documents(docs)
    index.insert_nodes(nodes)
    _query_engine = None
    print(f"  Indexed {len(nodes)} nodes from '{name}'")


def index_audio(path: str | Path):
    global _query_engine
    path = Path(path)
    name = path.name
    if _already_indexed(name):
        print(f"  Skipping '{name}' — already indexed")
        return
    print(f"  Transcribing with Docling ASR: '{name}'")
    result = _doc_converter.convert(str(path))
    md_text = result.document.export_to_markdown()
    if not md_text.strip():
        print(f"  Warning: empty transcript for '{name}'")
        return
    doc = Document(
        text=md_text,
        metadata={
            "file_name": name,
            "source_type": "audio",
            "source": str(path),
        },
    )
    nodes = _splitter.get_nodes_from_documents([doc])
    index.insert_nodes(nodes)
    _query_engine = None
    print(f"  Indexed {len(nodes)} nodes from '{name}' (audio transcript)")


def index_file(path: str | Path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        index_pdf(path)
    elif ext in IMAGE_EXTS:
        index_image(path)
    elif ext in AUDIO_EXTS:
        index_audio(path)
    else:
        print(f"  Unsupported file type: '{path.name}' (ext={ext})")


def index_directory(directory: str | Path):
    global _query_engine
    directory = Path(directory)
    all_files = [
        f
        for f in directory.rglob("*")
        if f.suffix.lower() in (PDF_EXTS | IMAGE_EXTS | AUDIO_EXTS)
    ]
    print(f"  Found {len(all_files)} supported files in '{directory}'")
    for f in all_files:
        index_file(f)
    _query_engine = None


# ─── Query engine ─────────────────────────────────────────────────────────────

_query_engine = None


def get_query_engine():
    global _query_engine
    if _query_engine is None:
        # vector_store_query_mode="hybrid" → Qdrant RRF fuses dense + sparse
        # results server-side before returning candidates to LlamaIndex reranker
        base_engine = index.as_query_engine(
            similarity_top_k=RETRIEVAL_K,
            sparse_top_k=SPARSE_K,
            vector_store_query_mode="hybrid",
            node_postprocessors=[reranker],
            response_mode="compact",
            text_qa_template=TEXT_QA_TEMPLATE,
        )
        hyde = HyDEQueryTransform(include_original=True)
        # _query_engine = TransformQueryEngine(base_engine, hyde)
        _query_engine = base_engine
    return _query_engine


def ask(query: str, verbose: bool = True) -> str:
    engine = get_query_engine()
    response = engine.query(query)
    if verbose:
        print(f"\n[Query] {query}")
        print(f"[Answer] {response}")
    return str(response)


# ─── Evaluation ───────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _token_f1(pred: str, ref: str) -> float:
    pred_tokens = _normalize(pred).split()
    ref_tokens = _normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


_sem_model = None


def _semantic_similarity(a: str, b: str) -> float:
    global _sem_model
    if _sem_model is None:
        _sem_model = SentenceTransformer(RERANK_MODEL, device="cpu")
    embs = _sem_model.encode([a, b], convert_to_tensor=True)
    cos = torch.nn.functional.cosine_similarity(
        embs[0].unsqueeze(0), embs[1].unsqueeze(0)
    )
    return cos.item()


def run_golden_eval(golden_path: str | Path, verbose: bool = True):
    data = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    golden = [(item["question"], item["answer"]) for item in data]
    all_em, all_f1, all_sim = [], [], []
    results = []

    for query, reference in golden:
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


# ─── Gradio UI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DOCS_DIR = Path("./docs")
    DOCS_DIR.mkdir(exist_ok=True)

    def chat(message: str, history: list) -> str:
        if not message.strip():
            return ""
        return ask(message, verbose=False)

    def upload_files(files, json_file) -> str:
        msgs = []
        if files:
            indexed, skipped, unsupported = [], [], []
            for f in files:
                p = Path(f.name)
                ext = p.suffix.lower()
                dest = DOCS_DIR / p.name
                shutil.copy(f.name, dest)
                if ext in PDF_EXTS | IMAGE_EXTS | AUDIO_EXTS:
                    before = _already_indexed(dest.name)
                    index_file(dest)
                    (skipped if before else indexed).append(dest.name)
                else:
                    unsupported.append(dest.name)
            if indexed:
                msgs.append(f"✅ Indexados: {', '.join(indexed)}")
            if skipped:
                msgs.append(f"⚠️ Já existiam: {', '.join(skipped)}")
            if unsupported:
                msgs.append(f"❌ Tipo não suportado: {', '.join(unsupported)}")

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
                chatbot = gr.Chatbot(
                    height=480, placeholder="Olá! Faça uma pergunta sobre o manual."
                )
                msg = gr.Textbox(placeholder="Ex: O que o Painel Edge faz?", scale=7)
                with gr.Row():
                    clear = gr.Button("🗑️ Limpar", size="sm")
                    retry = gr.Button("🔄 Tentar novamente", size="sm")
                    undo = gr.Button("↩️ Desfazer", size="sm")

                def respond(message, chat_history):
                    if not message.strip():
                        return "", chat_history
                    tuple_history = [
                        (chat_history[i]["content"], chat_history[i + 1]["content"])
                        for i in range(0, len(chat_history) - 1, 2)
                        if i + 1 < len(chat_history)
                    ]
                    bot_message = chat(message, tuple_history)
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": bot_message})
                    return "", chat_history

                def clear_chat():
                    return []

                def retry_last(chat_history):
                    if chat_history and len(chat_history) >= 2:
                        last_user_msg = chat_history[-2]["content"]
                        chat_history = chat_history[:-2]
                        tuple_history = [
                            (chat_history[i]["content"], chat_history[i + 1]["content"])
                            for i in range(0, len(chat_history) - 1, 2)
                            if i + 1 < len(chat_history)
                        ]
                        bot_message = chat(last_user_msg, tuple_history)
                        chat_history.append({"role": "user", "content": last_user_msg})
                        chat_history.append(
                            {"role": "assistant", "content": bot_message}
                        )
                    return chat_history

                def undo_last(chat_history):
                    if chat_history and chat_history[-1]["role"] == "assistant":
                        chat_history.pop()
                        if chat_history and chat_history[-1]["role"] == "user":
                            chat_history.pop()
                    return chat_history

                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                clear.click(clear_chat, None, chatbot)
                retry.click(retry_last, chatbot, chatbot)
                undo.click(undo_last, chatbot, chatbot)

            with gr.Tab("📄 Gerenciamento de arquivos"):
                gr.Markdown(
                    "**Tipos aceitos:** PDF · Imagens (PNG, JPG, TIFF, BMP, WEBP) · Áudio (MP3, WAV, M4A, OGG, FLAC) · Vídeo (MP4, MOV, AVI)"
                )
                file_upload = gr.File(
                    label="Documentos (PDF / Imagem / Áudio / Vídeo)",
                    file_types=[
                        ".pdf",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".tiff",
                        ".tif",
                        ".bmp",
                        ".webp",
                        ".mp3",
                        ".wav",
                        ".m4a",
                        ".ogg",
                        ".flac",
                        ".mp4",
                        ".mov",
                        ".avi",
                    ],
                    file_count="multiple",
                )
                json_upload = gr.File(
                    label="Golden Q&A (JSON)", file_types=[".json"], file_count="single"
                )
                upload_btn = gr.Button("📥 Enviar", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)
                upload_btn.click(
                    fn=upload_files,
                    inputs=[file_upload, json_upload],
                    outputs=upload_status,
                )
                reset_btn = gr.Button("🗑️ Resetar Qdrant", variant="stop")
                reset_status = gr.Textbox(label="Status reset", interactive=False)
                reset_btn.click(fn=reset_db, inputs=None, outputs=reset_status)

            with gr.Tab("📊 Avaliação Golden Q&A"):
                eval_btn = gr.Button("▶️ Executar avaliação", variant="primary")
                eval_output = gr.Markdown()
                eval_btn.click(fn=run_eval, inputs=None, outputs=eval_output)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="blue", font=gr.themes.GoogleFont("IBM Plex Sans")
        ),
    )
