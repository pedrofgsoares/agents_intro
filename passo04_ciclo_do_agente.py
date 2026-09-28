"""Passo 4 — O ciclo do agente.

O QUE VAIS APRENDER
    Um agente é, no essencial, isto:

        enquanto o modelo pedir ferramentas:
            executar as ferramentas
            devolver os resultados ao modelo
        quando ele responder com texto -> terminámos

    O modelo decide sozinho QUE ferramentas usar, em que ORDEM, e QUANDO parar.
    Nós só fornecemos as ferramentas e o ciclo. Esta é a ideia mais
    importante da formação: tudo o que vem a seguir são variações disto.

    Duas regras de segurança que aparecem aqui pela primeira vez:
      - um limite de passos (MAX_PASSOS), para o agente nunca ficar em ciclo infinito;
      - os erros das ferramentas são devolvidos ao modelo como texto, em vez
        de rebentarem o programa — assim o modelo pode corrigir-se.

COMO CORRER
    uv run passo04_ciclo_do_agente.py
    uv run passo04_ciclo_do_agente.py "a tua pergunta"

O QUE OBSERVAR
    - Cada "— passo N —" é uma volta do ciclo. Com a pergunta de exemplo, o
      modelo precisa de 2 voltas: primeiro multiplica, só depois pode somar
      (precisa do resultado da multiplicação).
    - Às vezes pede duas ferramentas no mesmo passo (hora + multiplicação).

EXPERIMENTA MUDAR
    - Uma pergunta que exija 3 ou 4 contas encadeadas. Quantos passos usa?
    - MAX_PASSOS = 1: o que acontece?
    - Pede uma ferramenta que não existe ("divide 10 por 2"): como reage o modelo ao ERRO?

PYTHON NOVO NESTE PASSO
    - range(1, N + 1): os números de 1 a N (para contar as voltas do ciclo).
    - try / except KeyError / except Exception: apanhar tipos de erro diferentes.
    - if __name__ == "__main__": o código que só corre quando executamos
      este ficheiro diretamente (e não quando outro ficheiro o importa).
    - sys.argv: o que se escreveu no terminal depois do nome do programa.
"""

import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["HF_BASE_URL"],
    api_key=os.environ["HF_API_KEY"],
)
MODEL = os.environ["HF_MODEL"]
EFFORT = os.getenv("HF_REASONING_EFFORT", "low")

# Travão de segurança: um agente pode entrar em ciclo infinito
# (ex.: uma ferramenta falha sempre e ele tenta de novo). Nunca omitas isto.
MAX_PASSOS = 10


# ---------------------------------------------------------------------------
# Ferramentas
# ---------------------------------------------------------------------------
def obter_hora_atual() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def multiplicar(a: float, b: float) -> float:
    return a * b


def somar(a: float, b: float) -> float:
    return a + b


FERRAMENTAS = {
    "obter_hora_atual": obter_hora_atual,
    "multiplicar": multiplicar,
    "somar": somar,
}


# Multiplicar e somar têm a mesma "forma" (dois números, a e b):
# esta função evita escrever duas vezes o mesmo dicionário.
def schema_dois_numeros(nome: str, descricao: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": nome,
            "description": descricao,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "obter_hora_atual",
            "description": "Devolve a data e hora atuais do sistema.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    schema_dois_numeros("multiplicar", "Multiplica dois números com precisão exata. Usa sempre para multiplicações."),
    schema_dois_numeros("somar", "Soma dois números com precisão exata. Usa sempre para somas."),
]


# ---------------------------------------------------------------------------
# Executar uma ferramenta pedida pelo modelo
# ---------------------------------------------------------------------------
def executar_ferramenta(nome: str, argumentos_json: str) -> str:
    # Se algo correr mal, NÃO rebentamos o programa: devolvemos o erro ao modelo
    # como texto. Assim ele pode perceber o problema e tentar de outra forma.
    try:
        funcao = FERRAMENTAS[nome]
        args = json.loads(argumentos_json or "{}")
        return str(funcao(**args))
    except KeyError:
        return f"ERRO: a ferramenta '{nome}' não existe."
    except Exception as e:
        return f"ERRO ao executar {nome}: {e}"


# ---------------------------------------------------------------------------
# O agente
# ---------------------------------------------------------------------------
def executar_agente(pergunta: str) -> str:
    messages = [
        {"role": "system", "content": (
            "És um assistente conciso. Respondes em português de Portugal. "
            "Usa as ferramentas disponíveis sempre que precisares de dados ou contas exatas."
        )},
        {"role": "user", "content": pergunta},
    ]

    # O CICLO DO AGENTE: pedir ao modelo → executar o que ele pedir → repetir
    for passo in range(1, MAX_PASSOS + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA, reasoning_effort=EFFORT
        )
        msg = response.choices[0].message
        messages.append(msg)

        # Condição de paragem: o modelo respondeu sem pedir ferramentas
        if not msg.tool_calls:
            return msg.content

        print(f"\n— passo {passo} —")
        for call in msg.tool_calls:
            resultado = executar_ferramenta(call.function.name, call.function.arguments)
            print(f"🔧 {call.function.name}({call.function.arguments}) → {resultado}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": resultado,
            })

    return f"(Parei ao fim de {MAX_PASSOS} passos sem resposta final.)"


if __name__ == "__main__":
    pergunta = " ".join(sys.argv[1:]) or (
        "Que horas são? Depois calcula 48213 vezes 9187 e soma 1000 ao resultado."
    )
    print("❓", pergunta)
    resposta = executar_agente(pergunta)
    print("\n💬", resposta)
