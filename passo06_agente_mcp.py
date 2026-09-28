"""Passo 6 — O agente como CLIENTE MCP.

O QUE VAIS APRENDER
    É o mesmo agente do passo 5 (while da conversa + for do agente), com uma
    diferença fundamental: as ferramentas NÃO estão neste ficheiro.

      1. O agente lança o servidor (mcp_servidor.py) como subprocesso
      2. Pergunta-lhe "que ferramentas tens?" (list_tools)
      3. Converte a resposta para o formato de tools da API OpenAI
      4. Quando o LLM pede uma ferramenta, reencaminha o pedido ao servidor (call_tool)

         LLM  ⇄  este agente  ⇄  servidor MCP (mcp_servidor.py)
                 (decide o fluxo)   (executa as ferramentas)

    Resultado: podes acrescentar ferramentas ao servidor sem tocar numa linha
    deste ficheiro — e o mesmo servidor serve também o Claude Code, o VS Code, etc.

COMO CORRER
    uv run passo06_agente_mcp.py
    (escreve 'sair' para terminar)

O QUE OBSERVAR
    - Ao arrancar, a lista "Ferramentas descobertas": o agente não as conhecia
      antes de perguntar ao servidor.
    - As chamadas aparecem com 🔌 (executadas no servidor) em vez de 🔧.
    - No passo06.log, o primeiro bloco mostra as ferramentas tal como o LLM as
      recebe. (O log é recriado em cada execução.)

EXPERIMENTA MUDAR
    - Acrescenta uma ferramenta ao mcp_servidor.py e volta a correr este passo.
    - Muda a docstring de uma ferramenta no servidor e vê a nova descrição aqui.

PYTHON NOVO NESTE PASSO
    - async def / await: este ficheiro é assíncrono porque o cliente MCP é
      assíncrono — fala com o servidor enquanto espera pelo LLM ou por ti.
      Lê "await x" como "espera por x, e entretanto deixa outras coisas avançar".
      Uma função async só corre com await (dentro de outra função async) ou
      com asyncio.run(...) (no arranque do programa).
    - async with: como o with, mas para coisas assíncronas (aqui: abrir e
      fechar a ligação ao servidor).
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import AsyncOpenAI

load_dotenv()

llm = AsyncOpenAI(
    base_url=os.environ["HF_BASE_URL"],
    api_key=os.environ["HF_API_KEY"],
)
MODEL = os.environ["HF_MODEL"]
EFFORT = os.getenv("HF_REASONING_EFFORT", "low")
MAX_PASSOS = 10

# Como lançar o servidor: com o mesmo Python deste ambiente virtual
SERVIDOR = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / "mcp_servidor.py")],
)
FICHEIRO_LOG = Path(__file__).parent / "passo06.log"


# ---------------------------------------------------------------------------
# Log: cada mensagem que entra no histórico (e portanto é enviada ao LLM,
# ou veio dele) é escrita no ficheiro, tal como a API a vê.
# ---------------------------------------------------------------------------
def registar_bloco(titulo: str, dados) -> None:
    with FICHEIRO_LOG.open("a", encoding="utf-8") as f:
        f.write(f"===== {datetime.now():%H:%M:%S} | {titulo}\n"
                f"{json.dumps(dados, ensure_ascii=False, indent=2)}\n\n")


def registar(mensagem, info: str = "") -> None:
    # As respostas do modelo são objetos da biblioteca openai; as nossas são dicts
    if not isinstance(mensagem, dict):
        mensagem = mensagem.model_dump(exclude_none=True)
    registar_bloco(mensagem["role"] + (f" | {info}" if info else ""), mensagem)


def mcp_para_openai(tool) -> dict:
    """Traduz a descrição de uma ferramenta MCP para o formato da API OpenAI.

    Os dois formatos usam JSON Schema para os parâmetros, por isso é só
    mudar os nomes dos campos.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


async def executar_ferramenta(mcp: Client, nome: str, argumentos_json: str) -> str:
    try:
        resultado = await mcp.call_tool(nome, json.loads(argumentos_json or "{}"))
    except Exception as e:  # ex.: JSON inválido vindo do LLM, servidor em baixo
        return f"ERRO: {e}"
    texto = "\n".join(c.text for c in resultado.content if c.type == "text")
    return f"ERRO: {texto}" if resultado.is_error else texto


async def turno_do_agente(mcp: Client, messages: list, tools: list) -> str:
    for _ in range(MAX_PASSOS):
        response = await llm.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, reasoning_effort=EFFORT
        )
        msg = response.choices[0].message
        messages.append(msg)
        u = response.usage
        registar(msg, f"tokens: enviados={u.prompt_tokens} recebidos={u.completion_tokens}")

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            resultado = await executar_ferramenta(mcp, call.function.name, call.function.arguments)
            resumo = resultado if len(resultado) < 80 else resultado[:77].replace("\n", " ") + "..."
            print(f"   🔌 {call.function.name}({call.function.arguments}) → {resumo}")
            mensagem_tool = {"role": "tool", "tool_call_id": call.id, "content": resultado}
            messages.append(mensagem_tool)
            registar(mensagem_tool, f"{call.function.name} (executada no servidor MCP)")

    return f"(Parei ao fim de {MAX_PASSOS} passos sem resposta final.)"


async def main():
    async with Client(SERVIDOR) as mcp:
        # Descoberta: o agente não sabia que ferramentas existiam até agora
        tools = [mcp_para_openai(t) for t in (await mcp.list_tools()).tools]
        print("Ligado ao servidor MCP. Ferramentas descobertas:")
        for t in tools:
            print(f"   • {t['function']['name']}: {t['function']['description']}")
        print(f"\nEscreve 'sair' para terminar. (log em {FICHEIRO_LOG.name})")

        messages = [
            {"role": "system", "content": (
                "És um assistente que ajuda o utilizador a aprender a construir agentes de IA. "
                "Respondes em português de Portugal, de forma concisa. "
                "Usa as ferramentas para consultar ficheiros em vez de adivinhar o seu conteúdo, "
                "e nunca apresentes um número que não tenha vindo de uma ferramenta."
            )},
        ]
        FICHEIRO_LOG.unlink(missing_ok=True)  # começar cada execução com um log limpo
        # As ferramentas vêm do servidor MCP e são enviadas ao LLM em cada pedido
        registar_bloco("tools (descobertas no servidor MCP)", tools)
        registar(messages[0])

        while True:
            try:
                # input() bloqueia; to_thread corre-o à parte para não congelar
                # a ligação ao servidor enquanto esperamos que escrevas
                pergunta = (await asyncio.to_thread(input, "\nTu: ")).strip()
            except EOFError:
                break
            if pergunta.lower() in {"sair", "exit", "quit"}:
                break
            if not pergunta:
                continue

            mensagem_user = {"role": "user", "content": pergunta}
            messages.append(mensagem_user)
            registar(mensagem_user)
            print(f"\nAgente: {await turno_do_agente(mcp, messages, tools)}")


if __name__ == "__main__":
    asyncio.run(main())
