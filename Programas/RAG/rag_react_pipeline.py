"""
ReAct Agent Pipeline — LangChain
=================================
Ciclo ReAct (Reasoning + Acting) sobre o pipeline RAG existente.
Importa objetos de rag_lang_pipeline.py sem duplicar lógica.

O agente decide autonomamente quando e o que buscar, podendo fazer
múltiplas buscas e reformular queries antes de dar a resposta final.

Uso:
  python rag_react_pipeline.py "sua pergunta aqui"
"""

import re
import os
import sys

# Garante import do pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)

import rag_lang_pipeline as rag
from langchain_ollama import ChatOllama

# LLM separado pro ReAct: mais contexto e tokens que o RAG simples
react_llm = ChatOllama(
    model=rag.OLLAMA_MODEL,
    temperature=0.1,
    keep_alive="-30m",
    num_ctx=8192,
    num_predict=1024,
    repeat_penalty=1.1,
)


# ─── Ferramentas do agente ────────────────────────────────────────────────────


@tool
def buscar_documentos(query: str) -> str:
    """Busca documentos relevantes na base de conhecimento.
    Use esta ferramenta para encontrar informacoes sobre qualquer assunto.
    A entrada deve ser a pergunta ou termos de busca."""
    docs = rag.retrieve_and_rerank(query)
    if not docs:
        return "Nenhum documento encontrado para essa busca."
    result_parts = []
    for i, d in enumerate(docs, 1):
        score = d.metadata.get("rerank_score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        # Trunca cada chunk pra não estourar o contexto
        content = d.page_content[:600]
        result_parts.append(f"[{i}] (score={score_str}) {content}")
    return "\n\n".join(result_parts)


@tool
def verificar_colecao(dummy: str = "") -> str:
    """Verifica quantos documentos estao indexados na base de conhecimento.
    Use antes de buscar para saber se ha documentos disponiveis.
    A entrada pode ser qualquer texto (sera ignorada)."""
    count = rag.vectorstore._collection.count()
    return f"Colecao '{rag.COLLECTION_NAME}': {count} documentos indexados."


# ─── Parser customizado (qwen3 emite <think> tags) ───────────────────────────


class QwenReActOutputParser(ReActSingleInputOutputParser):
    """Remove blocos <think>...</think> do qwen3 antes de parsear."""

    def parse(self, text: str) -> AgentAction | AgentFinish:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return super().parse(cleaned)


# ─── Prompt ReAct ─────────────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template(
    """\
Responda a pergunta do usuario usando as ferramentas disponiveis.
Voce so pode responder com base nos documentos recuperados.
Se nao encontrar a resposta, diga honestamente.
Responda no mesmo idioma da pergunta. Seja direto e conciso.

Voce tem acesso as seguintes ferramentas:

{tools}

Use EXATAMENTE o formato abaixo:

Question: a pergunta do usuario
Thought: raciocine sobre o que fazer
Action: a acao a tomar, deve ser uma de [{tool_names}]
Action Input: a entrada para a acao
Observation: o resultado da acao
... (Thought/Action/Action Input/Observation pode repetir N vezes)
Thought: I now know the final answer
Final Answer: a resposta final para a pergunta original

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)


# ─── Função principal ─────────────────────────────────────────────────────────


def ask_react(query: str, verbose: bool = False) -> dict:
    """Executa o ciclo ReAct para responder uma pergunta.

    Returns:
        dict com keys: answer, steps, error
    """
    tools = [buscar_documentos, verificar_colecao]

    agent = create_react_agent(
        llm=react_llm,
        tools=tools,
        prompt=REACT_PROMPT,
        output_parser=QwenReActOutputParser(),
        stop_sequence=["\nObservation"],
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=3,
        return_intermediate_steps=True,
        handle_parsing_errors=(
            "Erro de parsing. Use o formato correto:\n"
            "Thought: seu raciocinio\n"
            "Action: nome_da_ferramenta\n"
            "Action Input: entrada\n"
            "Ou para responder:\n"
            "Thought: I now know the final answer\n"
            "Final Answer: sua resposta"
        ),
        verbose=verbose,
    )

    try:
        result = executor.invoke({"input": query})
        answer = result.get("output", "")

        steps = []
        for action, observation in result.get("intermediate_steps", []):
            steps.append(
                {
                    "thought": action.log.strip(),
                    "action": action.tool,
                    "action_input": (
                        action.tool_input
                        if isinstance(action.tool_input, str)
                        else str(action.tool_input)
                    ),
                    "observation": str(observation)[:500],
                }
            )

        return {"answer": answer, "steps": steps, "error": None}

    except Exception as e:
        # Fallback pro RAG simples
        fallback_answer = rag.ask(query, verbose=verbose)
        return {
            "answer": fallback_answer,
            "steps": [],
            "error": f"ReAct falhou ({e}), usando RAG simples como fallback.",
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pergunta = " ".join(sys.argv[1:])
    else:
        pergunta = input("Pergunta: ").strip()

    if not pergunta:
        print("Nenhuma pergunta fornecida.")
        sys.exit(1)

    print(f"\n[ReAct] Pergunta: {pergunta}\n")
    resultado = ask_react(pergunta, verbose=True)

    print(f"\n{'='*60}")
    print(f"Resposta: {resultado['answer']}")
    if resultado["steps"]:
        print(f"\nPassos do raciocinio ({len(resultado['steps'])}):")
        for i, step in enumerate(resultado["steps"], 1):
            print(f"  [{i}] Acao: {step['action']}({step['action_input'][:50]})")
    if resultado["error"]:
        print(f"\nFallback: {resultado['error']}")
    print("=" * 60)
