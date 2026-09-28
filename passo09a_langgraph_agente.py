"""Passo 9a — O agente do passo 7, reescrito com LangChain/LangGraph.

O QUE VAIS APRENDER
    Mesmo prompt, mesmas ferramentas (o mesmo servidor mcp_email.py), mesma
    lógica: o LLM decide que ferramentas chamar e quando parar.

    Compara com o passo 7 o que DEIXÁMOS de escrever:
      - o ciclo do agente (for + tool_calls + role "tool")   -> create_agent
      - a tradução MCP -> formato OpenAI (mcp_para_openai)   -> MCPAdapter
      - executar_ferramenta e o tratamento de erros          -> MCPAdapter

    E o que GANHÁMOS sem esforço (middleware):
      - ModelRetryMiddleware:     repete a chamada ao LLM se falhar (ex: o erro
                                  "<|channel|>" que vimos na avaliação do passo 8)
      - ModelCallLimitMiddleware: o nosso MAX_PASSOS

    O preço: há mais "magia" escondida. Quando algo corre mal, é preciso saber
    o que a framework faz por baixo — que é exatamente o que aprendeste nos passos 1 a 7.

    Reutiliza do passo 7: o SYSTEM_PROMPT, a pasta de saída e a tabela de resumo.

COMO CORRER
    uv run passo09a_langgraph_agente.py          # todos os emails
    uv run passo09a_langgraph_agente.py 004 006  # só estes

O QUE OBSERVAR
    - O resultado é o mesmo tipo de tabela do passo 7 — o comportamento também.
    - Compara o tamanho deste ficheiro com o do passo 7: o ciclo desapareceu.
    - O passo09a.log mostra as mensagens tal como a LangChain as vê.
    - Para medir: uv run passo08_avaliacao.py --agente passo09a (≈ o mesmo que o 7).

EXPERIMENTA MUDAR
    - Tira o ModelRetryMiddleware e corre a avaliação: aparecem mais erros "E"?
    - Muda o limite do ModelCallLimitMiddleware para 3: o que acontece aos emails?

PYTHON NOVO NESTE PASSO
    - @asynccontextmanager + yield: uma função que prepara algo (ligar ao
      servidor), entrega-o a quem a usa com "async with", e no fim arruma
      (desliga). O yield é o ponto em que "entrega".
    - str | None: "um texto, ou nada" (None = ausência de valor).
"""

import asyncio
import json
import os
import shutil
import sys
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastmcp.client.transports import StdioTransport
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", category=LangChainBetaWarning)  # langchain.mcp ainda é beta
from langchain.mcp import MCPAdapter  # noqa: E402

from passo07_agente_email import PASTA_SAIDA, SYSTEM_PROMPT, mostrar_resumo  # noqa: E402

load_dotenv()

PASTA_PROJETO = Path(__file__).parent
FICHEIRO_LOG = PASTA_PROJETO / "passo09a.log"
MAX_PASSOS = 15


# ---------------------------------------------------------------------------
# Peças partilhadas com o passo 9b
# ---------------------------------------------------------------------------
def criar_modelo(modelo: str | None = None, effort: str | None = None) -> ChatOpenAI:
    # O ChatOpenAI fala com qualquer API compatível com a OpenAI, como o router da HF
    return ChatOpenAI(
        model=modelo or os.environ["HF_MODEL"],
        base_url=os.environ["HF_BASE_URL"],
        api_key=os.environ["HF_API_KEY"],
        reasoning_effort=effort or os.getenv("HF_REASONING_EFFORT", "low"),
    )


@asynccontextmanager
async def ligar_ferramentas(env: dict | None = None):
    """Lança o servidor mcp_email.py e devolve as suas ferramentas já como ferramentas LangChain."""
    transporte = StdioTransport(
        command=sys.executable,
        args=[str(PASTA_PROJETO / "mcp_email.py")],
        env=env or {**os.environ, "EMAIL_LOG_NIVEL": "CRITICAL"},
    )
    async with MCPAdapter(transporte) as adaptador:
        yield await adaptador.list_tools()


def texto_de(conteudo) -> str:
    """O conteúdo das mensagens LangChain pode ser texto ou uma lista de blocos."""
    if isinstance(conteudo, str):
        return conteudo
    return "\n".join(b.get("text", "") for b in conteudo if isinstance(b, dict))


def registar_bloco(ficheiro: Path | None, titulo: str, dados) -> None:
    if ficheiro is None:  # a avaliação corre sem log (execuções em paralelo misturavam-se)
        return
    with ficheiro.open("a", encoding="utf-8") as f:
        f.write(f"===== {datetime.now():%H:%M:%S} | {titulo}\n"
                f"{json.dumps(dados, ensure_ascii=False, indent=2, default=str)}\n\n")


def repetir_se(excecao: Exception) -> bool:
    # Por omissão, erros "400 Bad Request" não se repetem (seria o mesmo pedido,
    # o mesmo erro). Mas o gpt-oss às vezes GERA chamadas a ferramentas mal
    # formadas (nome com "<|channel|>", JSON inválido nos argumentos ou na
    # saída estruturada) e o
    # fornecedor devolve 400. São falhas aleatórias: repetir costuma resultar.
    mensagem = str(excecao)
    if any(erro in mensagem for erro in ("Tool call validation failed", "tool_use_failed", "output_parse_failed")):
        return True
    return getattr(excecao, "is_retryable", True)


# ---------------------------------------------------------------------------
# O agente
# ---------------------------------------------------------------------------
async def analisar(ferramentas, id_email: str, modelo=None, effort=None, log: Path | None = None) -> dict:
    agente = create_agent(
        criar_modelo(modelo, effort),
        ferramentas,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ModelRetryMiddleware(max_retries=2, retry_on=repetir_se),
            ModelCallLimitMiddleware(run_limit=MAX_PASSOS, exit_behavior="end"),
        ],
    )
    inicio = time.monotonic()
    resultado = await agente.ainvoke(
        {"messages": [{"role": "user", "content": f"Analisa o email com o id {id_email}."}]}
    )
    mensagens = resultado["messages"]

    registar_bloco(log, f"########## NOVA CONVERSA: email {id_email} ##########", "")
    registar_bloco(log, "system", SYSTEM_PROMPT)
    for m in mensagens:
        registar_bloco(log, m.type, m.model_dump(
            include={"content", "tool_calls", "tool_call_id", "name", "usage_metadata", "additional_kwargs"},
            exclude_none=True,
        ))

    # Traduzir as mensagens para o formato que a avaliação (passo 8) entende
    erros = {m.tool_call_id for m in mensagens
             if isinstance(m, ToolMessage) and texto_de(m.content).startswith("Error executing")}
    chamadas, tokens, passos = [], 0, 0
    for m in mensagens:
        if isinstance(m, AIMessage):
            passos += 1
            tokens += (m.usage_metadata or {}).get("total_tokens", 0)
            chamadas += [{"ferramenta": c["name"], "argumentos": c["args"], "erro": c["id"] in erros}
                         for c in m.tool_calls]
    return {"chamadas": chamadas, "passos": passos, "tokens": tokens,
            "segundos": round(time.monotonic() - inicio, 1),
            "conclusao": texto_de(mensagens[-1].content)}


async def main():
    FICHEIRO_LOG.unlink(missing_ok=True)
    shutil.rmtree(PASTA_SAIDA, ignore_errors=True)

    async with ligar_ferramentas() as ferramentas:
        todos = sorted(p.stem for p in (PASTA_PROJETO / "dados" / "caixa_ficticia").glob("*.eml"))
        ids = sys.argv[1:] or todos
        for id_email in ids:
            print(f"\n📧 Email {id_email}")
            r = await analisar(ferramentas, id_email, log=FICHEIRO_LOG)
            for c in r["chamadas"]:
                print(f"   🔌 {c['ferramenta']}({json.dumps(c['argumentos'], ensure_ascii=False)[:60]})")
            print(f"   💬 {r['conclusao'][:200]}")

    mostrar_resumo(ids, FICHEIRO_LOG.name)


if __name__ == "__main__":
    asyncio.run(main())
