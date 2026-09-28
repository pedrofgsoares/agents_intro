"""Servidor MCP com as ferramentas do projeto (usado pelo passo 6).

O QUE VAIS APRENDER
    MCP (Model Context Protocol) é um protocolo padrão para disponibilizar
    ferramentas a QUALQUER agente: o teu (passo 6), o Claude Code, o VS Code...
    Escreves as ferramentas uma vez; qualquer cliente MCP as pode descobrir e usar.
    É como uma tomada elétrica: qualquer aparelho com a ficha certa encaixa.

    Repara no que este ficheiro NÃO tem:
      - nenhum LLM, nenhuma API key -> o servidor só executa ferramentas
      - nenhum JSON Schema escrito à mão -> é gerado a partir dos type hints
        e da docstring de cada função (compara com o TOOLS_SCHEMA do passo 5)

    Comunicação: por "stdio" -> o cliente lança este script como subprocesso e
    fala com ele por stdin/stdout. Por isso, NUNCA uses print() aqui: iria
    misturar texto com as mensagens do protocolo e partir a comunicação.

COMO CORRER
    Normalmente não o corres tu: é o passo 6 que o lança.
    Para o testar sozinho (abre uma página web para chamar as ferramentas à mão;
    precisa do Node.js instalado):
        npx @modelcontextprotocol/inspector uv run mcp_servidor.py

O QUE OBSERVAR
    - As mesmas ferramentas do passo 5, agora num ficheiro à parte.
    - Cada ferramenta é só uma função com @mcp.tool() por cima.

EXPERIMENTA MUDAR
    - Acrescenta uma ferramenta (ex.: contar_palavras(texto: str) -> str) com
      @mcp.tool() e uma docstring. Corre o passo 6: ele descobre-a sozinho,
      sem mudares nada no passo 6.

PYTHON NOVO NESTE PASSO
    - Decoradores (@mcp.tool()): uma linha por cima de uma função que lhe
      acrescenta comportamento. Aqui: "regista esta função como ferramenta".
    - Docstrings (texto entre três aspas logo a seguir ao def): a descrição da função.
      O MCP usa-a como a descrição que o modelo lê.
    - raise ToolError("..."): lançar um erro de propósito, com uma mensagem.
"""

import ast
import operator
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

PASTA_PROJETO = Path(__file__).parent.resolve()
PROIBIDOS = {".env", ".venv", ".git", "__pycache__"}
MAX_CARACTERES = 8000

mcp = MCPServer(
    "ferramentas-do-projeto",
    instructions="Ferramentas para consultar os ficheiros deste projeto de aprendizagem e fazer contas exatas.",
)


# ---------------------------------------------------------------------------
# Funções auxiliares (não são ferramentas: não têm @mcp.tool)
# ---------------------------------------------------------------------------
_OPERADORES = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
}


def _avaliar(no):
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return no.value
    if isinstance(no, ast.BinOp) and type(no.op) in _OPERADORES:
        return _OPERADORES[type(no.op)](_avaliar(no.left), _avaliar(no.right))
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERADORES:
        return _OPERADORES[type(no.op)](_avaliar(no.operand))
    raise ToolError("expressão não permitida (só números e + - * / ** % //)")


def _caminho_seguro(caminho: str) -> Path:
    alvo = (PASTA_PROJETO / caminho).resolve()
    if not alvo.is_relative_to(PASTA_PROJETO):
        raise ToolError("acesso negado: fora da pasta do projeto")
    if any(parte in PROIBIDOS for parte in alvo.relative_to(PASTA_PROJETO).parts):
        raise ToolError("acesso negado: ficheiro/pasta protegido")
    return alvo


# ---------------------------------------------------------------------------
# Ferramentas: basta o decorador @mcp.tool()
# A docstring passa a ser a "description" que o LLM vê.
#
# Erros: uma ToolError chega ao modelo com a nossa mensagem (erro previsto).
# Qualquer outra exceção chega só como "Error executing tool X" — o detalhe
# fica no servidor, para não revelar informação interna por acidente.
# ---------------------------------------------------------------------------
@mcp.tool()
def obter_hora_atual() -> str:
    """Devolve a data e hora atuais."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def calcular(expressao: str) -> str:
    """Calcula uma expressão aritmética exata, ex: '(48213 * 9187) + 1000'. Usa sempre para contas."""
    try:
        return str(_avaliar(ast.parse(expressao, mode="eval").body))
    except (SyntaxError, ZeroDivisionError) as e:
        raise ToolError(f"expressão inválida: {e}")


@mcp.tool()
def listar_ficheiros(pasta: str = ".") -> str:
    """Lista os ficheiros de uma pasta do projeto. Caminho relativo à raiz; '.' para a raiz."""
    alvo = _caminho_seguro(pasta)
    if not alvo.is_dir():
        raise ToolError(f"a pasta '{pasta}' não existe")
    itens = sorted(
        p.name + ("/" if p.is_dir() else "")
        for p in alvo.iterdir()
        if p.name not in PROIBIDOS and not p.name.startswith(".")
    )
    return "\n".join(itens) or "(pasta vazia)"


@mcp.tool()
def ler_ficheiro(caminho: str) -> str:
    """Lê o conteúdo de um ficheiro de texto do projeto, ex: 'passo01_chamada_simples.py'."""
    alvo = _caminho_seguro(caminho)
    if not alvo.is_file():
        raise ToolError(f"o ficheiro '{caminho}' não existe")
    texto = alvo.read_text(encoding="utf-8")
    if len(texto) > MAX_CARACTERES:
        return texto[:MAX_CARACTERES] + f"\n\n[... cortado: ficheiro tem {len(texto)} caracteres]"
    return texto


# Nota: o escrever_nota do passo 5 ficou de fora de propósito. Ele usava input()
# para pedir autorização — mas aqui o stdin pertence ao protocolo! Num servidor
# MCP, a confirmação humana é responsabilidade do CLIENTE (o Claude Code, por
# exemplo, pergunta-te antes de executar ferramentas).


if __name__ == "__main__":
    mcp.run()  # por omissão: transporte stdio
