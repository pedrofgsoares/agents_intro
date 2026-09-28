"""Passo 1 — Uma chamada simples a um LLM.

O QUE VAIS APRENDER
    - Como o nosso programa fala com um modelo de linguagem (LLM): envia uma
      lista de mensagens e recebe uma resposta.
    - Que cada mensagem tem um papel ("role"): system, user ou assistant.
    - Que o prompt de sistema muda a forma como o modelo responde.
    Ainda NÃO é um agente — mas tudo o que vamos construir assenta nesta peça.

COMO CORRER
    uv run passo01_chamada_simples.py

O QUE OBSERVAR
    - A resposta vem no estilo pedido no prompt de sistema (um pirata!).
    - No fim aparece o número de tokens gastos: é assim que se mede o custo.
    - Corre duas vezes: a resposta muda. Um LLM não é determinístico.

EXPERIMENTA MUDAR
    - O texto do "system": pede outra personagem, ou "responde só com 10 palavras".
    - A pergunta do "user".
    - No .env, HF_MODEL para openai/gpt-oss-120b: a resposta fica melhor? Gasta mais?

PYTHON NOVO NESTE PASSO
    - Listas [ ... ]: uma sequência de coisas, por ordem.
    - Dicionários { "chave": valor }: pares nome → valor. Cada mensagem é um dicionário.
    - os.environ["NOME"]: lê uma configuração do .env (dá erro se não existir);
      os.getenv("NOME", "omissão") faz o mesmo mas usa um valor por omissão.
    - print(...): mostra texto no terminal.
    (Import, variáveis e f-strings: vê o passo 0 e o GLOSSARIO.md.)
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Lê o ficheiro .env e coloca as variáveis em os.environ
load_dotenv()

# O "client" é o objeto que sabe falar com o servidor do modelo.
# O router da Hugging Face fala o "dialeto" da API da OpenAI,
# por isso usamos o cliente da OpenAI apontado para outro endereço.
client = OpenAI(
    base_url=os.environ["HF_BASE_URL"],
    api_key=os.environ["HF_API_KEY"],
)

# Uma conversa é uma lista de mensagens, cada uma com um "role":
#   system    -> instruções de comportamento para o modelo
#   user      -> o que o utilizador escreve
#   assistant -> o que o modelo respondeu antes (histórico)
messages = [
    {"role": "system", "content": "Responde como um pirata maléfico. Respondes em português de Portugal."},
    {"role": "user", "content": "Em duas frases: o que é um agente de IA?"},
]

# O pedido ao modelo: qual modelo, que mensagens, e quanto deve "pensar"
response = client.chat.completions.create(
    model=os.environ["HF_MODEL"],
    messages=messages,
    reasoning_effort=os.getenv("HF_REASONING_EFFORT", "low"),
)

# A resposta traz várias coisas; o texto está em choices[0].message.content
print(response.choices[0].message.content)
# Um token é um pedaço de palavra. Paga-se (e há limites) por tokens enviados + recebidos.
print("\n--- tokens usados:", response.usage.total_tokens)
