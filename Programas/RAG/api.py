"""
FastAPI backend para o RAG Pipeline (LangChain).
Importa os objetos e funções de rag_lang_pipeline.py sem duplicar lógica.

Uso:
  python Programas/RAG/api.py
  # Acesse http://localhost:8000/docs para o Swagger
"""

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Garante que o diretório do pipeline está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_lang_pipeline as rag
from rag_react_pipeline import ask_react

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="RAG LangChain API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    query: str


class SourceChunk(BaseModel):
    content: str
    source: str
    rerank_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class IndexTextRequest(BaseModel):
    text: str
    doc_id: str = "manual"


class StatusResponse(BaseModel):
    collection_name: str
    document_count: int


class EvalResult(BaseModel):
    question: str
    reference: str
    prediction: str
    exact_match: int
    token_f1: float
    semantic_sim: float


class EvalResponse(BaseModel):
    results: list[EvalResult]
    total: int
    exact_match_pct: float
    avg_token_f1: float
    avg_semantic_sim: float


class MessageResponse(BaseModel):
    message: str


class ReactStep(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class ReactAskResponse(BaseModel):
    answer: str
    steps: list[ReactStep]
    fallback_used: bool
    error: str | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """Faz uma pergunta ao RAG e retorna resposta + fontes."""
    docs = rag.retrieve_and_rerank(req.query)
    context = rag._format_context(docs)
    prompt = rag.PROMPT_TEMPLATE.format_messages(
        context=context, question=req.query
    )
    # Tenta até 3 vezes (gemma3:4b ocasionalmente retorna vazio)
    answer = ""
    for attempt in range(3):
        result = rag.llm.invoke(prompt)
        answer = (
            result.content.strip()
            if hasattr(result, "content")
            else str(result).strip()
        )
        if answer:
            break

    sources = [
        SourceChunk(
            content=d.page_content[:500],
            source=Path(
                str(d.metadata.get("file_path") or d.metadata.get("source", "?"))
            ).name,
            rerank_score=d.metadata.get("rerank_score"),
        )
        for d in docs
    ]
    return AskResponse(answer=answer, sources=sources)


@app.post("/upload", response_model=MessageResponse)
def upload_pdf(file: UploadFile = File(...)):
    """Faz upload de um PDF e indexa no ChromaDB."""
    count_before = rag.vectorstore._collection.count()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(file.file.read())
        tmp.close()
        rag.index_pdf(tmp.name)
    finally:
        os.unlink(tmp.name)
    count_after = rag.vectorstore._collection.count()
    added = count_after - count_before
    return MessageResponse(
        message=f"PDF '{file.filename}' indexado com sucesso ({added} chunks)."
    )


@app.post("/index-text", response_model=MessageResponse)
def index_text(req: IndexTextRequest):
    """Indexa um texto bruto."""
    count_before = rag.vectorstore._collection.count()
    rag.index_text(req.text, req.doc_id)
    count_after = rag.vectorstore._collection.count()
    added = count_after - count_before
    return MessageResponse(
        message=f"Texto '{req.doc_id}' indexado ({added} chunks)."
    )


@app.get("/collection/status", response_model=StatusResponse)
def collection_status():
    """Retorna informações da coleção."""
    return StatusResponse(
        collection_name=rag.COLLECTION_NAME,
        document_count=rag.vectorstore._collection.count(),
    )


@app.post("/collection/reset", response_model=MessageResponse)
def collection_reset():
    """Limpa a coleção e recria vazia."""
    count = rag.vectorstore._collection.count()
    rag.vectorstore.reset_collection()
    return MessageResponse(
        message=f"Coleção limpa ({count} documentos removidos)."
    )


@app.post("/eval", response_model=EvalResponse)
def run_eval():
    """Executa avaliação golden Q&A e retorna métricas."""
    golden_path = rag.GOLDEN_PATH
    raw_results = rag.run_golden_eval(golden_path, verbose=False)

    results = []
    for r in raw_results:
        results.append(
            EvalResult(
                question=r["question"],
                reference=r["reference"],
                prediction=r["prediction"],
                exact_match=r["metrics"]["exact_match"],
                token_f1=r["metrics"]["token_f1"],
                semantic_sim=r["metrics"]["semantic_sim"],
            )
        )

    total = len(results)
    em_sum = sum(r.exact_match for r in results)
    f1_avg = sum(r.token_f1 for r in results) / total if total else 0
    sim_avg = sum(r.semantic_sim for r in results) / total if total else 0

    return EvalResponse(
        results=results,
        total=total,
        exact_match_pct=round(100 * em_sum / total, 1) if total else 0,
        avg_token_f1=round(f1_avg, 4),
        avg_semantic_sim=round(sim_avg, 4),
    )


@app.post("/ask-react", response_model=ReactAskResponse)
def ask_react_endpoint(req: AskRequest):
    """Faz uma pergunta usando o agente ReAct (raciocinio passo a passo)."""
    result = ask_react(req.query, verbose=False)
    steps = [ReactStep(**s) for s in result["steps"]]
    return ReactAskResponse(
        answer=result["answer"],
        steps=steps,
        fallback_used=result["error"] is not None,
        error=result["error"],
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
