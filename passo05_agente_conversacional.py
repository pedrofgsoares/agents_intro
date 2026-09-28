"""Passo 5 — Agente conversacional com ferramentas úteis (e seguras).

O QUE VAIS APRENDER
    Junta tudo o que vimos:
      - passo 2: ciclo EXTERIOR de conversa (while), com memória em `messages`
      - passo 4: ciclo INTERIOR do agente (for com MAX_PASSOS)

         ┌─ while (conversa contigo) ───────────────────────┐
         │   tu perguntas                                   │
         │   ┌─ for (o agente trabalha) ─────────────────┐  │
         │   │  modelo → ferramentas → modelo → ...      │  │
         │   └───────────────────────────────────────────┘  │
         │   o agente responde                              │
         └──────────────────────────────────────────────────┘

    Ferramentas novas:
      - calcular:        avalia expressões matemáticas (sem usar eval!)
      - listar_ficheiros / ler_ficheiro: o agente pode explorar ESTE projeto
      - escrever_nota:   cria ficheiros em notas/, mas só com a TUA autorização

    Segurança — um agente com acesso a ficheiros precisa de limites:
      1. Só pode ver ficheiros dentro da pasta do projeto
      2. Nunca pode ler segredos (.env)
      3. Ações que alteram coisas pedem confirmação a um humano ("human-in-the-loop")

COMO CORRER
    uv run passo05_agente_conversacional.py
    No chat: 'sair' para terminar, '/historico' para ver o tamanho da memória.

O QUE OBSERVAR
    - Pergunta "o que faz o passo 3?": o agente lista e lê ficheiros sozinho.
    - Pede "lê o ficheiro .env": a ferramenta recusa, e o agente explica porquê.
    - Pede "faz um resumo do passo 4 e guarda-o numa nota": aparece ✋ a pedir
      autorização. Experimenta recusar.
    - Abre o passo05.log: é EXATAMENTE o que foi enviado ao modelo e recebido
      dele, incluindo os resultados das ferramentas. (Recriado em cada execução.)

EXPERIMENTA MUDAR
    - Tenta convencer o agente a ler ../ (fora do projeto). A proteção está no
      código (_caminho_seguro), não no prompt: não depende de o modelo obedecer.
    - Retira a pergunta de autorização em escrever_nota. Porque é má ideia?
    - Muda o prompt de sistema para "és um tutor exigente que faz perguntas".

PYTHON NOVO NESTE PASSO
    - Path (biblioteca pathlib): representa caminhos de ficheiros;
      pasta / "ficheiro" junta caminhos.
    - with ... as f: abre um ficheiro e garante que é fechado no fim.
    - isinstance(x, tipo): verifica de que tipo é um valor.
    - Funções que começam por _ (ex.: _avaliar): por convenção, "uso interno".
"""

import ast
import json
import operator
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["HF_BASE_URL"],
    api_key=os.environ["HF_API_KEY"],
)
MODEL = os.environ["HF_MODEL"]
EFFORT = os.getenv("HF_REASONING_EFFORT", "low")
MAX_PASSOS = 10

PASTA_PROJETO = Path(__file__).parent.resolve()
PASTA_NOTAS = PASTA_PROJETO / "notas"
PROIBIDOS = {".env", ".venv", ".git", "__pycache__"}
MAX_CARACTERES = 8000  # não enviar ficheiros enormes ao modelo (custa tokens)
FICHEIRO_LOG = PASTA_PROJETO / "passo05.log"


# ---------------------------------------------------------------------------
# Log: cada mensagem que entra no histórico (e portanto é enviada ao LLM,
# ou veio dele) é escrita no ficheiro, tal como a API a vê.
# ---------------------------------------------------------------------------
def registar(mensagem, info: str = "") -> None:
    # As respostas do modelo são objetos da biblioteca openai; as nossas são dicts
    if not isinstance(mensagem, dict):
        mensagem = mensagem.model_dump(exclude_none=True)
    cabecalho = f"===== {datetime.now():%H:%M:%S} | {mensagem['role']}"
    if info:
        cabecalho += f" | {info}"
    with FICHEIRO_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{cabecalho}\n{json.dumps(mensagem, ensure_ascii=False, indent=2)}\n\n")


# ---------------------------------------------------------------------------
# Ferramentas
# ---------------------------------------------------------------------------
def obter_hora_atual() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Calculadora segura: percorre a "árvore" da expressão e só aceita números
# e operadores aritméticos. Usar eval() seria perigoso: o modelo (ou alguém
# que o manipule) poderia pedir para executar código arbitrário.
_OPERADORES = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
}


def _avaliar(no):
    # Função recursiva: para avaliar "2 + 3 * 4" avalia primeiro as partes
    # (2, e 3 * 4) chamando-se a si própria, e depois junta-as.
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return no.value
    if isinstance(no, ast.BinOp) and type(no.op) in _OPERADORES:
        return _OPERADORES[type(no.op)](_avaliar(no.left), _avaliar(no.right))
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERADORES:
        return _OPERADORES[type(no.op)](_avaliar(no.operand))
    raise ValueError("expressão não permitida (só números e + - * / ** % //)")


def calcular(expressao: str) -> str:
    return str(_avaliar(ast.parse(expressao, mode="eval").body))


def _caminho_seguro(caminho: str) -> Path:
    """Converte o caminho pedido num caminho real e verifica se é permitido."""
    alvo = (PASTA_PROJETO / caminho).resolve()
    # resolve() transforma "../../etc/passwd" no caminho verdadeiro,
    # por isso esta verificação apanha tentativas de "fugir" da pasta.
    if not alvo.is_relative_to(PASTA_PROJETO):
        raise PermissionError("acesso negado: fora da pasta do projeto")
    if any(parte in PROIBIDOS for parte in alvo.relative_to(PASTA_PROJETO).parts):
        raise PermissionError("acesso negado: ficheiro/pasta protegido")
    return alvo


def listar_ficheiros(pasta: str = ".") -> str:
    alvo = _caminho_seguro(pasta)
    itens = sorted(
        p.name + ("/" if p.is_dir() else "")
        for p in alvo.iterdir()
        if p.name not in PROIBIDOS and not p.name.startswith(".")
    )
    return "\n".join(itens) or "(pasta vazia)"


def ler_ficheiro(caminho: str) -> str:
    texto = _caminho_seguro(caminho).read_text(encoding="utf-8")
    if len(texto) > MAX_CARACTERES:
        return texto[:MAX_CARACTERES] + f"\n\n[... cortado: ficheiro tem {len(texto)} caracteres]"
    return texto


def escrever_nota(nome: str, conteudo: str) -> str:
    alvo = (PASTA_NOTAS / nome).resolve()
    if not alvo.is_relative_to(PASTA_NOTAS):
        raise PermissionError("só é permitido escrever dentro de notas/")

    # Human-in-the-loop: ações com efeitos pedem autorização
    print(f"\n   ✋ O agente quer escrever em notas/{nome}:")
    print("   " + conteudo[:300].replace("\n", "\n   ") + ("..." if len(conteudo) > 300 else ""))
    if input("   Autorizas? [s/N] ").strip().lower() != "s":
        return "O utilizador RECUSOU a escrita. Não tentes novamente sem lhe perguntar."

    PASTA_NOTAS.mkdir(exist_ok=True)
    alvo.write_text(conteudo, encoding="utf-8")
    return f"Nota guardada em notas/{nome}"


FERRAMENTAS = {
    "obter_hora_atual": obter_hora_atual,
    "calcular": calcular,
    "listar_ficheiros": listar_ficheiros,
    "ler_ficheiro": ler_ficheiro,
    "escrever_nota": escrever_nota,
}


def _schema(nome: str, descricao: str, /, **params: str) -> dict:
    """Atalho para criar schemas onde todos os parâmetros são texto obrigatório."""
    return {
        "type": "function",
        "function": {
            "name": nome,
            "description": descricao,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS_SCHEMA = [
    _schema("obter_hora_atual", "Devolve a data e hora atuais."),
    _schema("calcular", "Calcula uma expressão aritmética exata. Usa sempre para contas.",
            expressao="Expressão Python aritmética, ex: '(48213 * 9187) + 1000'"),
    _schema("listar_ficheiros", "Lista os ficheiros de uma pasta do projeto.",
            pasta="Caminho relativo à raiz do projeto; '.' para a raiz"),
    _schema("ler_ficheiro", "Lê o conteúdo de um ficheiro de texto do projeto.",
            caminho="Caminho relativo à raiz do projeto, ex: 'passo01_chamada_simples.py'"),
    _schema("escrever_nota", "Guarda uma nota em texto/markdown na pasta notas/. O utilizador terá de aprovar.",
            nome="Nome do ficheiro, ex: 'resumo.md'", conteudo="Conteúdo completo da nota"),
]


def executar_ferramenta(nome: str, argumentos_json: str) -> str:
    try:
        funcao = FERRAMENTAS[nome]
        args = json.loads(argumentos_json or "{}")
        return str(funcao(**args))
    except KeyError:
        return f"ERRO: a ferramenta '{nome}' não existe."
    except Exception as e:
        return f"ERRO ao executar {nome}: {e}"


# ---------------------------------------------------------------------------
# Ciclo INTERIOR: o agente trabalha até ter uma resposta
# ---------------------------------------------------------------------------
def turno_do_agente(messages: list) -> str:
    for _ in range(MAX_PASSOS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA, reasoning_effort=EFFORT
        )
        msg = response.choices[0].message
        messages.append(msg)
        u = response.usage
        registar(msg, f"tokens: enviados={u.prompt_tokens} recebidos={u.completion_tokens}")

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            resultado = executar_ferramenta(call.function.name, call.function.arguments)
            resumo = resultado if len(resultado) < 80 else resultado[:77].replace("\n", " ") + "..."
            print(f"   🔧 {call.function.name}({call.function.arguments}) → {resumo}")
            mensagem_tool = {"role": "tool", "tool_call_id": call.id, "content": resultado}
            messages.append(mensagem_tool)
            registar(mensagem_tool, call.function.name)

    return f"(Parei ao fim de {MAX_PASSOS} passos sem resposta final.)"


# ---------------------------------------------------------------------------
# Ciclo EXTERIOR: a conversa contigo
# ---------------------------------------------------------------------------
def main():
    # Criado FORA dos dois ciclos: a memória sobrevive entre perguntas,
    # incluindo os resultados das ferramentas.
    messages = [
        {"role": "system", "content": (
            "És um assistente que ajuda o utilizador a aprender a construir agentes de IA. "
            "Respondes em português de Portugal, de forma concisa. "
            "Tens acesso aos ficheiros deste projeto: usa as ferramentas para os consultar "
            "em vez de adivinhar o seu conteúdo."
        )},
    ]
    FICHEIRO_LOG.unlink(missing_ok=True)  # começar cada execução com um log limpo
    registar(messages[0])
    print(f"Agente pronto. Escreve 'sair' para terminar. (log em {FICHEIRO_LOG.name})")

    while True:
        try:
            pergunta = input("\nTu: ").strip()
        except EOFError:
            break
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if pergunta == "/historico":
            print(f"   [{len(messages)} mensagens no histórico]")
            continue
        if not pergunta:
            continue

        mensagem_user = {"role": "user", "content": pergunta}
        messages.append(mensagem_user)
        registar(mensagem_user)
        print(f"\nAgente: {turno_do_agente(messages)}")


if __name__ == "__main__":
    main()
