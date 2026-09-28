"""Passo 2 — Chat com memória.

O QUE VAIS APRENDER
    - O modelo NÃO se lembra de nada entre pedidos. Cada pedido começa do zero.
    - A "memória" de um chat é só a lista `messages`, que vai crescendo.
      Em cada volta: juntamos a mensagem do utilizador, enviamos TUDO ao
      modelo, e juntamos também a resposta dele à lista.
    - Por isso, quanto mais longa a conversa, mais tokens cada pedido gasta.

COMO CORRER
    uv run passo02_chat_com_memoria.py
    (escreve 'sair' para terminar)

O QUE OBSERVAR
    - Diz o teu nome e, umas perguntas depois, pergunta "como me chamo?".
    - Repara na linha "[mensagens no histórico | tokens enviados]": os tokens
      enviados crescem a cada volta, porque enviamos a conversa TODA.

EXPERIMENTA MUDAR
    - Põe um # antes da linha `messages.append({"role": "assistant", ...})`
      (passo 3 do ciclo). O modelo continua a saber o que ele próprio disse?
    - Muda a personalidade no prompt de sistema.

PYTHON NOVO NESTE PASSO
    - while True: repete o bloco para sempre (até um `break`).
    - break / continue: sair do ciclo / saltar para a volta seguinte.
    - input("..."): espera que o utilizador escreva uma linha no terminal.
    - lista.append(x): acrescenta x ao fim da lista.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["HF_BASE_URL"],
    api_key=os.environ["HF_API_KEY"],
)

# A conversa começa só com as instruções de sistema; o resto vai-se juntando
messages = [
    {"role": "system", "content": "És um assistente simpático e conciso. Respondes em português de Portugal."},
]

while True:
    try:
        pergunta = input("\nTu: ").strip()  # .strip() tira espaços no início e no fim
    except EOFError:  # Ctrl+D
        break
    if pergunta.lower() in {"sair", "exit", "quit"}:
        break
    if not pergunta:  # linha vazia: volta a perguntar
        continue

    # 1) Guardar a mensagem do utilizador no histórico
    messages.append({"role": "user", "content": pergunta})

    # 2) Enviar o histórico COMPLETO ao modelo
    response = client.chat.completions.create(
        model=os.environ["HF_MODEL"],
        messages=messages,
        reasoning_effort=os.getenv("HF_REASONING_EFFORT", "low"),
    )
    resposta = response.choices[0].message.content

    # 3) Guardar a resposta do modelo no histórico — sem isto, ele "esquece"
    messages.append({"role": "assistant", "content": resposta})

    print(f"\nModelo: {resposta}")
    print(f"   [mensagens no histórico: {len(messages)} | tokens enviados: {response.usage.prompt_tokens}]")
