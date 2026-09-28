"""Passo 3 — Tool calling (uma ronda).

O QUE VAIS APRENDER
    - Os LLMs erram em contas e não sabem as horas. A solução: dar-lhes
      FERRAMENTAS (funções nossas) que eles podem pedir para usar.
    - O modelo NÃO executa código. Ele apenas PEDE: "chama a função X com
      estes argumentos". Quem executa é o nosso programa, que depois devolve
      o resultado ao modelo.
    - O modelo só conhece as ferramentas pela descrição que lhe damos.

    Fluxo desta ronda:
      1. Enviamos a pergunta + a descrição das ferramentas disponíveis
      2. O modelo responde com um pedido de ferramenta (tool_calls)
      3. Executamos a função Python correspondente
      4. Enviamos o resultado ao modelo (role "tool")
      5. O modelo usa esse resultado para dar a resposta final

COMO CORRER
    uv run passo03_tool_calling.py

O QUE OBSERVAR
    - As linhas 🔧 mostram o que o modelo PEDIU; as linhas ↳ o que NÓS executámos.
    - A multiplicação fica exata: 48213 × 9187 = 442932831.
    - Às vezes o modelo quer mais ferramentas depois de ver os resultados, e
      o programa não consegue continuar (⚠️). Esse problema é resolvido no passo 4.

EXPERIMENTA MUDAR
    - Apaga a descrição da ferramenta "multiplicar" (description vazia): o modelo ainda a usa?
    - Pergunta algo que não precise de ferramentas ("qual a capital de França?").
    - Acrescenta uma ferramenta "somar" (função + entrada em FERRAMENTAS + descrição).

PYTHON NOVO NESTE PASSO
    - def nome(a: float) -> float: define uma função; ": float" e "-> float"
      são só indicações do tipo de valor (não mudam o funcionamento).
    - return: o valor que a função devolve.
    - for x in lista: repete o bloco para cada elemento da lista.
    - json.loads(texto): converte texto JSON em dicionário Python.
    - funcao(**args): chama a função passando cada par do dicionário como argumento.
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["HF_BASE_URL"],
    api_key=os.environ["HF_API_KEY"],
)
# Nomes em MAIÚSCULAS: por convenção, valores que não mudam durante o programa
MODEL = os.environ["HF_MODEL"]
EFFORT = os.getenv("HF_REASONING_EFFORT", "low")


# ---------------------------------------------------------------------------
# As ferramentas: funções Python normais
# ---------------------------------------------------------------------------
def obter_hora_atual() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def multiplicar(a: float, b: float) -> float:
    return a * b


# Mapa nome -> função, para sabermos o que executar quando o modelo pedir
FERRAMENTAS = {
    "obter_hora_atual": obter_hora_atual,
    "multiplicar": multiplicar,
}

# A descrição das ferramentas que o modelo "vê" (JSON Schema).
# O modelo decide QUANDO usar cada uma com base no nome e na descrição —
# por isso descrições claras são muito importantes.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "obter_hora_atual",
            "description": "Devolve a data e hora atuais do sistema.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiplicar",
            "description": "Multiplica dois números com precisão exata. Usa sempre para multiplicações.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Primeiro número"},
                    "b": {"type": "number", "description": "Segundo número"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


messages = [
    {"role": "system", "content": "És um assistente conciso. Respondes em português de Portugal."},
    {"role": "user", "content": "Que horas são? E quanto é 48213 vezes 9187?"},
]

# --- 1) Primeira chamada: pergunta + ferramentas ---------------------------
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=TOOLS_SCHEMA, reasoning_effort=EFFORT
)
msg = response.choices[0].message

# Se o modelo respondeu com texto em vez de pedir ferramentas, não há mais nada a fazer
if not msg.tool_calls:
    print("O modelo respondeu sem usar ferramentas:\n", msg.content)
    raise SystemExit

# --- 2) O modelo pediu ferramentas ------------------------------------------
# Guardamos o pedido dele no histórico (obrigatório: as respostas "tool"
# têm de vir a seguir à mensagem do assistant que as pediu)
messages.append(msg)

# O modelo pode pedir várias ferramentas de uma vez: tratamos cada pedido
for call in msg.tool_calls:
    nome = call.function.name
    args = json.loads(call.function.arguments or "{}")  # os argumentos vêm como texto JSON
    print(f"🔧 O modelo pediu: {nome}({args})")

    # --- 3) NÓS executamos a função ----------------------------------------
    resultado = FERRAMENTAS[nome](**args)
    print(f"   ↳ resultado: {resultado}")

    # --- 4) Devolvemos o resultado, ligado ao pedido pelo tool_call_id ------
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": str(resultado),
    })

# --- 5) Segunda chamada: o modelo já tem os resultados ----------------------
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=TOOLS_SCHEMA, reasoning_effort=EFFORT
)
final = response.choices[0].message

if final.tool_calls:
    # O modelo, depois de ver os resultados, quer AINDA MAIS ferramentas.
    # Com uma única ronda não conseguimos responder — é isto que o passo 4 resolve.
    pedidos = [f"{c.function.name}({c.function.arguments})" for c in final.tool_calls]
    print("\n⚠️  O modelo ainda quer mais ferramentas:", ", ".join(pedidos))
    print("   Uma ronda só não chega. Precisamos de um CICLO → passo 4.")
else:
    print("\n💬 Resposta final:", final.content)
