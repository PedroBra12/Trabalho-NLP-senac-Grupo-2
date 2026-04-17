"""
Streamlit frontend para o RAG Pipeline.
Comunica-se com o backend FastAPI (api.py) via HTTP.

Uso:
  streamlit run Programas/RAG/app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def api_get(path: str):
    """GET request à API."""
    return requests.get(f"{API_URL}{path}", timeout=30)


def api_post(path: str, **kwargs):
    """POST request à API."""
    return requests.post(f"{API_URL}{path}", timeout=600, **kwargs)


def check_api():
    """Verifica se a API está no ar."""
    try:
        r = api_get("/collection/status")
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
)

# ─── Session state ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RAG LangChain")

    # Status da API
    api_online = check_api()
    if api_online:
        st.success("API conectada")
    else:
        st.error(
            "API offline. Inicie o backend:\n\n"
            "`python Programas/RAG/api.py`"
        )

    st.divider()

    # ── Modo ──────────────────────────────────────────────────────────────────
    st.header("Modo")
    modo_rag = st.radio(
        "Selecione o modo:",
        ["RAG Simples", "ReAct Agent"],
        index=0,
        help=(
            "RAG Simples: busca + resposta direta. "
            "ReAct: agente com raciocinio passo a passo."
        ),
    )

    st.divider()

    # ── Upload PDF ────────────────────────────────────────────────────────────
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Selecione um PDF", type=["pdf"])
    if uploaded_file and st.button("Enviar PDF", use_container_width=True):
        if not api_online:
            st.error("API offline.")
        else:
            with st.spinner("Indexando PDF..."):
                try:
                    r = api_post(
                        "/upload",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    )
                    if r.status_code == 200:
                        st.success(r.json()["message"])
                    else:
                        st.error(f"Erro {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Erro: {e}")

    st.divider()

    # ── Indexar texto ─────────────────────────────────────────────────────────
    st.header("Indexar Texto")
    text_input = st.text_area("Cole o texto aqui", height=100)
    text_id = st.text_input("ID do documento", value="manual")
    if st.button("Indexar Texto", use_container_width=True):
        if not api_online:
            st.error("API offline.")
        elif not text_input.strip():
            st.warning("Texto vazio.")
        else:
            with st.spinner("Indexando..."):
                try:
                    r = api_post(
                        "/index-text",
                        json={"text": text_input, "doc_id": text_id},
                    )
                    if r.status_code == 200:
                        st.success(r.json()["message"])
                    else:
                        st.error(f"Erro {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Erro: {e}")

    st.divider()

    # ── Coleção ───────────────────────────────────────────────────────────────
    st.header("Coleção")
    if api_online:
        try:
            status = api_get("/collection/status").json()
            st.metric("Documentos indexados", status["document_count"])
            st.caption(f"Coleção: `{status['collection_name']}`")
        except Exception:
            st.warning("Erro ao obter status.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Atualizar", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Limpar", type="secondary", use_container_width=True):
            if api_online:
                try:
                    r = api_post("/collection/reset")
                    if r.status_code == 200:
                        st.success(r.json()["message"])
                    else:
                        st.error(f"Erro {r.status_code}")
                except Exception as e:
                    st.error(f"Erro: {e}")

    st.divider()

    # ── Avaliacao ─────────────────────────────────────────────────────────────
    st.header("Avaliacao Golden Q&A")
    if st.button("Executar Avaliacao", use_container_width=True):
        if not api_online:
            st.error("API offline.")
        else:
            with st.spinner("Executando avaliacao (pode demorar)..."):
                try:
                    r = api_post("/eval")
                    if r.status_code == 200:
                        data = r.json()
                        st.metric("Exact Match", f"{data['exact_match_pct']}%")
                        st.metric("Avg Token F1", f"{data['avg_token_f1']:.4f}")
                        st.metric("Avg Semantic Sim", f"{data['avg_semantic_sim']:.4f}")
                        st.caption(f"Total: {data['total']} perguntas")

                        with st.expander("Detalhes por pergunta"):
                            for r_item in data["results"]:
                                st.markdown(f"**Q:** {r_item['question']}")
                                st.markdown(f"**Resposta:** {r_item['prediction']}")
                                st.markdown(f"**Referencia:** {r_item['reference']}")
                                st.markdown(
                                    f"EM={r_item['exact_match']} | "
                                    f"F1={r_item['token_f1']:.4f} | "
                                    f"Sim={r_item['semantic_sim']:.4f}"
                                )
                                st.divider()
                    else:
                        st.error(f"Erro {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Erro: {e}")


# ─── Area principal: Chat ─────────────────────────────────────────────────────

st.title("RAG Pipeline - Assistente")

# Renderizar historico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Fontes utilizadas"):
                for i, src in enumerate(msg["sources"], 1):
                    score = src.get("rerank_score")
                    score_str = f"{score:.4f}" if score is not None else "N/A"
                    st.markdown(
                        f"**[{i}]** `{src['source']}` (score: {score_str})"
                    )
                    st.caption(src["content"][:300])
        if msg.get("mode") == "react" and msg.get("steps"):
            with st.expander("Raciocinio do Agente (ReAct)"):
                for i, step in enumerate(msg["steps"], 1):
                    st.markdown(f"**Passo {i}**")
                    st.markdown(f"**Pensamento:** {step['thought']}")
                    st.markdown(
                        f"**Acao:** `{step['action']}` "
                        f"com entrada: `{step['action_input'][:100]}`"
                    )
                    st.markdown(
                        f"**Observacao:** {step['observation'][:300]}"
                    )
                    st.divider()

# Input do chat
if query := st.chat_input("Faca sua pergunta..."):
    # Mostra mensagem do usuario
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Consulta a API
    with st.chat_message("assistant"):
        if not api_online:
            answer = "API offline. Inicie o backend com `python Programas/RAG/api.py`."
            st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        elif modo_rag == "RAG Simples":
            with st.spinner("Consultando pipeline RAG..."):
                try:
                    r = api_post("/ask", json={"query": query})
                    if r.status_code == 200:
                        data = r.json()
                        answer = data["answer"]
                        sources = data["sources"]

                        st.markdown(answer)
                        if sources:
                            with st.expander("Fontes utilizadas"):
                                for i, src in enumerate(sources, 1):
                                    score = src.get("rerank_score")
                                    score_str = (
                                        f"{score:.4f}" if score is not None else "N/A"
                                    )
                                    st.markdown(
                                        f"**[{i}]** `{src['source']}` "
                                        f"(score: {score_str})"
                                    )
                                    st.caption(src["content"][:300])

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    else:
                        err = f"Erro da API: {r.status_code}"
                        st.error(err)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err}
                        )
                except requests.exceptions.ConnectionError:
                    err = "Nao foi possivel conectar a API."
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
                except Exception as e:
                    err = f"Erro: {e}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )

        else:
            # ── ReAct Agent ───────────────────────────────────────────────
            with st.spinner("Agente ReAct raciocinando..."):
                try:
                    r = api_post("/ask-react", json={"query": query})
                    if r.status_code == 200:
                        data = r.json()
                        answer = data["answer"]
                        steps = data.get("steps", [])

                        st.markdown(answer)

                        if data.get("fallback_used"):
                            st.warning(data["error"])

                        if steps:
                            with st.expander(
                                "Raciocinio do Agente (ReAct)", expanded=True
                            ):
                                for i, step in enumerate(steps, 1):
                                    st.markdown(f"**Passo {i}**")
                                    st.markdown(f"**Pensamento:** {step['thought']}")
                                    st.markdown(
                                        f"**Acao:** `{step['action']}` "
                                        f"com entrada: `{step['action_input'][:100]}`"
                                    )
                                    st.markdown(
                                        f"**Observacao:** {step['observation'][:300]}"
                                    )
                                    st.divider()

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "steps": steps,
                                "mode": "react",
                            }
                        )
                    else:
                        err = f"Erro da API: {r.status_code}"
                        st.error(err)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err}
                        )
                except requests.exceptions.ConnectionError:
                    err = "Nao foi possivel conectar a API."
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
                except Exception as e:
                    err = f"Erro: {e}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
